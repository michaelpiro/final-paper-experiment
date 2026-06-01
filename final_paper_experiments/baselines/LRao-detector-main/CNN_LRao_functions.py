import torch
import numpy as np
import matplotlib.pyplot as plt
import scipy
import numpy as np
from sklearn.model_selection import RepeatedKFold
from tqdm.auto import tqdm
device = "cuda" if torch.cuda.is_available() else "cpu"


class Trafo(torch.nn.Module):
    """
    Transformation (CNN) that maximizes LFI at its output.
    """
    def __init__(self,config):
        super().__init__()
        if "dim_inp_train" in config:
            self.dim_inp = config["dim_inp_train"]
        else:
            self.dim_inp = config["dim_inp"]
        self.da = config["da"]
        bias = False
        layers = []
        layers.append(torch.nn.Conv1d(1, config["n_hidden_channels"], config["filt_size"], bias=bias, padding="same"))
        layers.append(torch.nn.Tanh())
        for _ in range(config["n_conv_layers"]-2):
            layers.append(torch.nn.Conv1d(config["n_hidden_channels"], config["n_hidden_channels"], config["filt_size"], bias=bias, padding="same"))
            layers.append(torch.nn.Tanh())
        layers.append(torch.nn.Conv1d(config["n_hidden_channels"], 1, config["filt_size"], bias=bias, padding="same"))
        layers.append(torch.nn.Flatten())
        self.filt = torch.nn.Sequential(*layers)

        self.filt.apply(init_weights)
        self.psd_method_params = config["psd_method_params"]
    

    def forward(self,x):
        """
        Apply transformation.
        """
        return self.filt(x[:,None,:])

    
    def lfi_diag_dft(self,x,puffer_sides,H=None):
        """
        Estimate the LFI contained in Trafo(x) with respect to additive signals specified by the columns in H using the DFT method.
        """
        if H==None:
            H = torch.eye(self.dim_inp,dtype=torch.float32,device=x.device)
        Pyy = self.est_psd(self.forward(x))
        lfi_diag = torch.zeros(H.shape[1],device=x.device)
        dMu_ = []
        for i in range(H.shape[1]):
            ds = self.da*H[:,i].reshape(1,-1)
            if puffer_sides>0:
                ds[0,:puffer_sides] = 0
                ds[0,-puffer_sides:] = 0
            x1 = x.clone()
            x1 -= ds
            x2 = x.clone()
            x2 += ds
            dmu = torch.mean((self.forward(x2)-self.forward(x1))/(2*self.da),dim=0)
            dMu = torch.fft.fft(dmu,norm="ortho")
            dMu_.append(dMu.detach())
            # Pyy_ = (self.est_psd(self.forward(x2))+self.est_psd(self.forward(x1)))/2
            lfi_diag[i] = torch.sum(torch.abs(dMu)**2/Pyy)
        return lfi_diag#, dMu_, Pyy.detach()
    
    def lfi_diag_autocorr(self,x,puffer_sides,H=None):
        """
        Estimate the LFI contained in Trafo(x) with respect to additive signals specified by the columns in H using the DFT method.
        """
        if H==None:
            H = torch.eye(self.dim_inp,dtype=torch.float32,device=x.device)
        _, Ryy = est_autocorr(self.forward(x))
        u,s,v = torch.linalg.svd(Ryy)
        s_inv = torch.zeros(s.shape,device=s.device,dtype=torch.float32)
        cutoff = 1e-3
        s_inv[s>cutoff] = 1/s[s>cutoff]
        Ryy_inv = v.T@torch.diag(s_inv)@u.T
        lfi_diag = torch.zeros(H.shape[1],device=x.device)
        for i in range(H.shape[1]):
            ds = self.da*H[:,i].reshape(1,-1)
            if puffer_sides>0:
                ds[0,:puffer_sides] = 0
                ds[0,-puffer_sides:] = 0
            x1 = x.clone()
            x1 -= ds
            x2 = x.clone()
            x2 += ds
            dmu = torch.mean((self.forward(x2)-self.forward(x1))/(2*self.da),dim=0)
            lfi_diag[i] = dmu@Ryy_inv@dmu
        return lfi_diag
    
    def est_psd(self,x):
        """
        Estimate PSD from data.
        """
        match self.psd_method_params["name"]:
            case "periodogram":
                return est_psd_averaged_periodogram(x)
            case "yule":
                return est_psd_ar_yule_walker(x,self.psd_method_params["order"])

    def set_statistics_autocorr(self,x,H=None):
        """
        Compute and set statistics (mean, gradient mean, inv cov mat, LFI) of Trafo(x).
        """
        if H==None:
            H = torch.eye(self.dim_inp,dtype=torch.float32,device=x.device)
        with torch.no_grad():
            y = self.forward(x)
            Ryy_inv = torch.inverse(est_autocorr(y)[1])
            self.mu0 = 0#torch.mean(y,dim=0,keepdim=True)
            self.dmu0 = torch.zeros(y.shape[1],H.shape[1],device=x.device)
            for i in range(H.shape[1]):
                x1 = x.clone()
                x1 -= self.da*H[:,i].reshape(1,-1)
                x2 = x.clone()
                x2 += self.da*H[:,i].reshape(1,-1)
                self.dmu0[:,i] = torch.mean((self.forward(x2)-self.forward(x1))/(2*self.da),dim=0)
        self.inv_cov = Ryy_inv
        self.j0 = self.dmu0.T@Ryy_inv@self.dmu0
        self.j0_inv = torch.inverse(self.j0)

    def set_statistics_dft(self,x,H=None):
        """
        Compute and set statistics (mean, gradient mean, inv cov mat, LFI) of Trafo(x).
        """
        if H==None:
            H = torch.eye(self.dim_inp,dtype=torch.float32,device=x.device)
        with torch.no_grad():
            y = self.forward(x)
            self.Pyy = self.est_psd(y)
            self.mu0 = 0#torch.mean(y,dim=0,keepdim=True)
            self.dmu0 = torch.zeros(y.shape[1],H.shape[1],device=x.device)
            for i in range(H.shape[1]):
                ds = self.da*H[:,i].reshape(1,-1)
                x1 = x.clone()
                x1 -= ds
                x2 = x.clone()
                x2 += ds
                self.dmu0[:,i] = torch.mean((self.forward(x2)-self.forward(x1))/(2*self.da),dim=0)
        dft_mat = torch.fft.fft(torch.eye(y.shape[1],device=x.device),norm="ortho")
        self.inv_cov = torch.real(torch.conj(dft_mat.T)@(dft_mat/self.Pyy.reshape(-1,1)))
        self.dMu0 = torch.fft.fft(self.dmu0,dim=0,norm="ortho")
        self.j0 = torch.real(torch.conj(self.dMu0).T@(self.dMu0/self.Pyy.reshape(-1,1)))
        self.j0_inv = torch.inverse(self.j0)
    
    def g(self,x):
        """
        Compute pressimistic score function.
        """
        return self.dmu0.T@self.inv_cov@(self.forward(x)-self.mu0).T

    def estimate(self,x,H=None):
        """
        Compute LBLUE estimate.
        """
        if H==None:
            j0_inv = self.j0_inv
            gx = self.g(x).T
        else:
            j0_inv = torch.inverse(H.T@self.j0@H)
            gx = self.g(x).T@H
        with torch.no_grad():
            return j0_inv@gx.T
        
    def estimate_norm(self,x,H=None):
        """
        Compute LBLUE estimate.
        """
        if H==None:
            j0_inv = self.j0_inv
            gx = self.g(x).T
        else:
            j0_inv = torch.inverse(H.T@self.j0@H)
            gx = self.g(x).T@H
        with torch.no_grad():
            return torch.tensor(scipy.linalg.sqrtm(j0_inv.cpu()),device=x.device)@gx.T
    
    def detect(self,x,H=None):
        """
        Compute LRao test statistic.
        """
        with torch.no_grad():
            if H==None:
                j0_inv = self.j0_inv
                gx = self.g(x).T
            else:
                j0_inv = torch.inverse(H.T@self.j0@H)
                gx = self.g(x).T@H
            return torch.concat([gx[i].reshape(1,-1)@j0_inv@gx[i].reshape(-1,1) for i in range(len(x))])

    def calc_test_statistics(self,x,H,theta,H0=True):
        """
        Returns test statistics (t0,t1) computed by LRao.
        """
        theta = (torch.tensor(theta.clone()).reshape(-1,1)).to(x.device)
        s = (H@theta).T
        t1 = self.detect(x+s)
        if H0:
            t0 = self.detect(x)
            return t0,t1
        else:
            return t1
    

def est_autocorr(x):
    """
    Nonparametric PSD estimate by averaging the periodogram.
    """
    x_four = torch.fft.fft(x,n=2*x.shape[1],dim=1)
    # s_four = x_four*torch.conj(x_four)
    r_xx = torch.fft.ifft(torch.abs(x_four)**2,dim=1)/x.shape[1]
    r_xx = torch.mean(torch.real(r_xx),dim=0)
    r_xx = r_xx[:x.shape[1]]*torch.bartlett_window(2*x.shape[1],device=x.device)[x.shape[1]:]
    return r_xx,pytorch_toeplitz(r_xx.reshape(1,-1))

def est_psd_averaged_periodogram(x):
    """
    Nonparametric PSD estimate by averaging the periodogram.
    """
    return torch.mean(torch.abs(torch.fft.fft(x,dim=1,norm="ortho"))**2,dim=0)

def est_psd_ar_yule_walker(x,p):
    """
    Autoregressive Parametric PSD estimate using Yule-Walker method.
    """
    a, sigma2 = yule_walker(x, p)
    w = 2*np.pi*torch.arange(x.shape[1],device=x.device).reshape(1,-1)/x.shape[1]
    psda = sigma2/(torch.abs(1-torch.sum(a.reshape(-1,1)*torch.exp(-1j*torch.arange(1,len(a)+1,device=x.device).reshape(-1,1)*w),dim=0))**2)
    return torch.tensor(psda.clone(),dtype=torch.complex64)

def yule_walker(x,order):
    """
    Solve Yule-Walker equations for autoregressive coefficients and noise variance.
    """
    r = torch.zeros((len(x),order+1),dtype=torch.float64,device=x.device)
    r[:,0] = torch.sum(x**2,dim=1)
    for k in range(1, order+1):
        r[:,k] = torch.sum(x[:,0:-k]*x[:,k:],dim=1)
    r = torch.mean(r,dim=0)/x.shape[1]
    R = pytorch_toeplitz(r[:-1].reshape(1,-1))
    a = torch.linalg.solve(R, r[1:])
    sigma2 = r[0] - (r[1:]*a).sum()
    return a, sigma2
    
def pytorch_toeplitz(V):
    """
    Construct Toeplitz matrix from given vector.
    """
    d = V.shape[1]
    A = V.unsqueeze(1).unsqueeze(2)
    A_nofirst_flipped = torch.flip(A[:, :, :, 1:], dims=[3]) 
    A_concat = torch.concatenate([A_nofirst_flipped, A], dim=3) 
    unfold = torch.nn.Unfold(kernel_size=(1, d))
    T = unfold(A_concat)
    T = torch.flip(T, dims=[2])
    return T.squeeze()

def reshape_data(data,dim_inp):
    """
    Reshape data sequence to [n_sequences,dim_inp].
    """
    return data[:(len(data)//dim_inp)*dim_inp].reshape(-1,dim_inp)
    
def train_data_sim(model,H,H_test,config,noise_gen):
    """
    Train model until stopping condition is reached.
    """
    match(config["optim"]):
        case "SGD":
            optimizer = torch.optim.SGD(model.parameters(), lr=config["lr"],weight_decay=config["weight_decay"])
        case "AdamW":
            optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"],weight_decay=config["weight_decay"])#
        case "Adam":
            optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"],weight_decay=config["weight_decay"])
    model.train()
    lfi = []
    for i in tqdm(range(config["max_iterations"])):
        for _ in range(config["iter_batch"]):
            data = noise_gen.gen_noise_sequences(config["dim_inp_train"],config["batch_size"])
            data = data.to(device)
            lfi_diag = model.lfi_diag_dft(data,config["puffer_sides"],H)
            if i%10==0:
                with torch.no_grad():
                    data = noise_gen.gen_noise_sequences(config["dim_inp_val"],config["batch_size"])
                    data = data.to(device)
                    lfi.append(model.lfi_diag_dft(data,0,H_test).cpu().detach())
                    print(f"lfi {lfi[-1]}")
            lfi_diag_mean = torch.mean(lfi_diag)
            loss = -lfi_diag_mean/config["iter_batch"]
            loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    data_stats = noise_gen.gen_noise_sequences(config["dim_inp_val"],config["batch_size"])
    data_stats = data_stats.to(device)
    model.eval()
    model.set_statistics_dft(data_stats.detach(),H_test)
    return lfi, model

def train_nested_cv_val(model,data_train,data_val,H,config):
    """
    Train model until stopping condition is reached.
    """
    match(config["optim"]):
        case "SGD":
            optimizer = torch.optim.SGD(model.parameters(), lr=config["lr"],weight_decay=config["weight_decay"])
        case "AdamW":
            optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"],weight_decay=config["weight_decay"])
        case "Adam":
            optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"],weight_decay=config["weight_decay"])
    model.train()
    lfi_val = []
    patience = 3
    early_stopper = EarlyStopper(patience=patience, min_delta=0)
    for i in range(config["max_iterations"]):
        lfi_diag_mean = torch.mean(model.lfi_diag_dft(data_train,config["puffer_sides"],H))
        loss = -lfi_diag_mean
        loss.backward()
        with torch.no_grad():
            lfi_val.append(torch.mean(model.lfi_diag_dft(data_val,0,H)).cpu())
        optimizer.step()
        optimizer.zero_grad()
        if early_stopper.early_stop(-lfi_val[-1]):
            break
    return max(lfi_val), np.argmax(lfi_val)

def train_nested_cv_test(model,data_train,data_test,H,amplitudes,config,epochs):
    """
    Train and test model for given hyperparameters.
    """
    match(config["optim"]):
        case "SGD":
            optimizer = torch.optim.SGD(model.parameters(), lr=config["lr"],weight_decay=config["weight_decay"])
        case "AdamW":
            optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"],weight_decay=config["weight_decay"])
        case "Adam":
            optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"],weight_decay=config["weight_decay"])
    model.train()
    lfi = []
    for i in range(epochs):
        lfi_diag_mean = torch.mean(model.lfi_diag_dft(data_train,config["puffer_sides"],H))
        lfi.append(lfi_diag_mean.cpu().detach())
        loss = -lfi_diag_mean
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    model.eval()
    model.set_statistics_dft(data_train.detach(),H)
    t0 = model.detect(data_test).cpu().squeeze()
    t1 = [None]*len(amplitudes)
    lambda_nc = torch.zeros(len(amplitudes),dtype=torch.float32)
    for i in range(len(amplitudes)):
        phases = 2*torch.pi*torch.rand((H.shape[1]//2,1),dtype=torch.float32,device=H.device)
        theta = torch.zeros((H.shape[1],1),device=H.device)
        theta[:len(theta)//2] = amplitudes[i]*torch.cos(phases)
        theta[len(theta)//2:] = -amplitudes[i]*torch.sin(phases)
        t1[i] = model.calc_test_statistics(data_test.detach(),H,theta,H0=False).cpu().squeeze()
        lambda_nc[i] = (theta.T@model.j0@theta).cpu().squeeze()
    return t0,torch.stack(t1,dim=0),lambda_nc, lfi
    
def nested_cv(data,H,amplitudes,config_list,n_splits_o,n_repeats_o,n_splits_i,n_repeats_i,n_repeats_phase_net):
    """
    Nested cross validation procedure.
    """
    t0 = []
    t1 = []
    lambda_nc = []
    lfi = []
    config_best = []
    epochs_best = []
    rkf = RepeatedKFold(n_splits=n_splits_o, n_repeats=n_repeats_o)
    for train_index, test_index in (pbar:=tqdm(rkf.split(data),total=n_repeats_o*n_splits_o,leave=False)):
        pbar.set_description("outer CV")
        lfi_val = []
        epochs_max_mean = []
        for config in (pbar_config:=tqdm(config_list,leave=False)):
            pbar_config.set_description("hyper params")
            lfi_val_ = []
            epochs_max = []
            rkf = RepeatedKFold(n_splits=n_splits_i, n_repeats=n_repeats_i)
            for train_index_i, val_index in (pbar:=tqdm(rkf.split(data[train_index]),total=n_repeats_i*n_splits_i,leave=False)):
                pbar.set_description("inner CV")
                model = Trafo(config)
                model = model.to(device=data.device)
                lfi_val_max, epoch_max = train_nested_cv_val(model,data[train_index_i],data[val_index],H,config)
                lfi_val_.append(lfi_val_max)
                epochs_max.append(epoch_max)
            lfi_val.append(np.mean(lfi_val_))
            epochs_max_mean.append(round(np.mean(epochs_max)))
        config_best_ = config_list[np.argmax(lfi_val)]
        config_best.append(config_best_)
        epochs_best_ = epochs_max_mean[np.argmax(lfi_val)]
        epochs_best.append(epochs_best_)
        for i in range(n_repeats_phase_net):
            model = Trafo(config_best_)
            model = model.to(device=data.device)
            t0_,t1_,lambda_nc_, lfi_ = train_nested_cv_test(model,data[train_index],data[test_index],H,amplitudes,config_best_,epochs_best_)
            t0.append(t0_)
            t1.append(t1_)
            lambda_nc.append(lambda_nc_)
            lfi.append(lfi_)
    return torch.concat(t0), torch.concat(t1,dim=1), torch.mean(torch.stack(lambda_nc),dim=0), lfi, config_best


def reference_detection(data,H,amplitudes,n_splits_o,n_repeats_o,n_repeats_phase):
    """
    Evaluate reference detection methods.
    """

    def test_statistics(reference_fun):
        t0 = []
        t1 = []
        rkf = RepeatedKFold(n_splits=n_splits_o, n_repeats=n_repeats_o)
        for train_index, test_index in rkf.split(data):
            t0.append(reference_fun(data[train_index],data[test_index],H).cpu().squeeze())
            t1_temp = [None]*len(amplitudes)
            for i in range(len(amplitudes)):
                t1_temp[i] = []
                for j in range(n_repeats_phase):
                    phases = 2*torch.pi*torch.rand((H.shape[1]//2,1),dtype=torch.float32,device=H.device)
                    theta = torch.zeros((H.shape[1],1),device=H.device)
                    theta[:len(theta)//2] = amplitudes[i]*torch.cos(phases)
                    theta[len(theta)//2:] = -amplitudes[i]*torch.sin(phases)
                    t1_temp[i].append(reference_fun(data[train_index],data[test_index]+(H@theta).T,H).cpu().squeeze())
                t1_temp[i] = torch.concat(t1_temp[i])
            t1.append(torch.stack(t1_temp,dim=0))
        t0 = torch.concat(t0)
        t1 = torch.concat(t1,dim=1)
        return t0, t1

    t0_RaoCGN, t1_RaoCGN = test_statistics(reference_RaoCGN)
    t0_RaoCGN_clipped, t1_RaoCGN_clipped = test_statistics(reference_RaoCGN_clipped)
    t0_RaoCLaplace, t1_RaoCLaplace = test_statistics(reference_RaoCLaplace)
    
    return t0_RaoCGN,t1_RaoCGN,t0_RaoCGN_clipped,t1_RaoCGN_clipped,t0_RaoCLaplace,t1_RaoCLaplace

def reference_RaoCGN(data_train,data_test,H):
    """
    Prewhitening + Rao detector for IID Gaussian noise. Equivalent to GLRT under Gaussian noise assumption.
    """
    dft_mat = torch.fft.fft(torch.eye(data_train.shape[1],device=data_train.device),norm="ortho")
    Pxx = torch.mean(torch.abs(torch.fft.fft(data_train,dim=1,norm="ortho"))**2,dim=0)
    inv_cov = torch.real(torch.conj(dft_mat.T)@(dft_mat/Pxx.reshape(-1,1)))
    mu = torch.mean(data_train,dim=0)
    gx = H.T@inv_cov@(data_test).T#-mu
    j0_inv = torch.inverse(H.T@inv_cov@H)
    return torch.concat([gx[:,i].reshape(1,-1)@j0_inv@gx[:,i].reshape(-1,1) for i in range(len(data_test))])

def reference_RaoCGN_clipped(data_train,data_test,H):
    """
    Prewhitening + heuristic Rao detector with limiter function non-linearity.
    """
    clip_lim = 3
    dft_mat = torch.fft.fft(torch.eye(data_train.shape[1],device=data_train.device),norm="ortho")
    Pxx = torch.median(torch.abs(torch.fft.fft(data_train,dim=1,norm="ortho"))**2,dim=0)[0]
    inv_cov = torch.real(torch.conj(dft_mat.T)@(dft_mat/Pxx.reshape(-1,1)))
    inv_cov_sqrt = torch.real(torch.conj(dft_mat.T)@(dft_mat/torch.sqrt(Pxx.reshape(-1,1))))
    gx = H.T@inv_cov_sqrt@clip_data(inv_cov_sqrt@(data_test).T,clip_lim,-clip_lim)
    j0_inv = torch.inverse(H.T@inv_cov@H)
    return torch.concat([gx[:,i].reshape(1,-1)@j0_inv@gx[:,i].reshape(-1,1) for i in range(len(data_test))])

# def reference_RaoWGN(data_test,H):
#     j_inv = torch.inverse(H.T@H)
#     return torch.concat([data_test[i].reshape(1,-1)@H@j_inv@H.T@data_test[i].reshape(-1,1) for i in range(len(data_test))])

# def reference_RaoWLaplace(data_test,H):
#     j_inv = torch.inverse(H.T@H)
#     return torch.concat([torch.sign(data_test[i]).reshape(1,-1)@H@j_inv@H.T@torch.sign(data_test[i]).reshape(-1,1) for i in range(len(data_test))])

def reference_RaoCLaplace(data_train,data_test,H):
    """
    Prewhitening + Rao detector for IID Laplace noise.
    """
    dft_mat = torch.fft.fft(torch.eye(data_train.shape[1],device=data_train.device),norm="ortho")
    Pxx = torch.mean(torch.abs(torch.fft.fft(data_train,dim=1,norm="ortho"))**2,dim=0)
    inv_cov_sqrt = torch.real(torch.conj(dft_mat.T)@(dft_mat/torch.sqrt(Pxx).reshape(-1,1)))
    data_test_ = (inv_cov_sqrt@data_test.T).T
    H_ = inv_cov_sqrt@H
    gx = H_.T@(torch.sign(data_test_)).T
    j0_inv = torch.inverse(H_.T@H_)
    return torch.concat([gx[:,i].reshape(1,-1)@j0_inv@gx[:,i].reshape(-1,1) for i in range(len(data_test_))])

def clip_data(x,lim_upper,lim_lower):
    """
    Limiter function.
    """
    y = x.clone()
    y[y>lim_upper] = lim_upper
    y[y<lim_lower] = lim_lower
    return y

def H_multi_harmonic(K,psi0,dim_inp):
    """
    Observation matrix for multi-harmonic signal.
    """
    n = torch.arange(dim_inp).reshape(-1,1)
    wk = 2*torch.pi*psi0*torch.arange(1,K+1).reshape(1,-1)
    H = torch.zeros(dim_inp,2*K)
    H[:,:K] = torch.cos(wk*n)
    H[:,K:] = torch.sin(wk*n)
    return H

def init_weights(m):
    """
    Initialize weights of conv1d layers using Xavier uniform technique (appropriate for tanh activations).
    """
    if isinstance(m, torch.nn.Conv1d):
        torch.nn.init.xavier_uniform_(m.weight, gain=torch.nn.init.calculate_gain("tanh"))
        # m.bias.data.fill_(0.0)

def auroc(fpr,tpr,fpr_max=1):
    """
    Compute AUROC FPR and TPR.
    """
    delta_fpr = torch.diff(fpr[fpr<=fpr_max])
    return torch.sum(tpr[fpr<=fpr_max][:-1]*delta_fpr).cpu().numpy()

def calc_roc_statistics(t0,t1):
    """
    Compute ROC statistics (TPR, FPR, Threshold) for given histograms (t0,t1).
    """
    upper_lim = torch.max(torch.max(t0,t1))
    lower_lim = torch.min(torch.min(t0,t1))
    gamma = torch.linspace(lower_lim-1e-1,upper_lim+1e-1,1000,device=t0.device)
    tp = torch.zeros(len(gamma),device=t0.device)
    tn = torch.zeros(len(gamma),device=t0.device)
    fp = torch.zeros(len(gamma),device=t0.device)
    fn = torch.zeros(len(gamma),device=t0.device)
    for i in range(len(gamma)):
        tp[i] = torch.sum(t1>gamma[i])
        tn[i] = torch.sum(t0<gamma[i])
        fp[i] = torch.sum(t0>gamma[i])
        fn[i] = torch.sum(t1<gamma[i])
    tpr = tp/(tp+fn)
    fpr = fp/(fp+tn)
    return tpr, fpr



class StudentT_noise_generator:
    """
    Generator for student-t noise passed through linear filter.
    """
    def __init__(self,nu,filt_coeffs):
        self.m = torch.distributions.studentT.StudentT(df=nu)
        self.filt_coeffs = filt_coeffs
    def gen_noise_sequences(self,sequence_len,batch_size):
        u = self.m.sample((batch_size,sequence_len))
        w = torch.tensor(scipy.signal.lfilter(self.filt_coeffs[1],self.filt_coeffs[0],u),dtype=torch.float32)
        return w

class EarlyStopper:
    """
    Returns true if validation_loss doesn't improve for patience iterations. min_delta sets margin for counting an increase in loss.
    """
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

def gen_list_models(config):
    """
    Returns list of dictionaries from dictionary of lists.
    """
    mesh = np.meshgrid(*config.values())
    model_list = [dict() for _ in range(len(mesh[0].flatten()))]
    for i in range(len(model_list)):
        for j,key in enumerate(config):
            model_list[i][key] = mesh[j].flatten()[i].copy()
    return model_list
