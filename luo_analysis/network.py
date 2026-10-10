"""Small trainable 1D CNN, NumPy only; SAME padding, ReLU, GAP, softmax."""
import numpy as np


def softmax(z):
    e = np.exp(z-z.max(axis=1,keepdims=True))
    return e/e.sum(axis=1,keepdims=True)


def windows(x, kernel, stride):
    out_length = (x.shape[1]+stride-1)//stride
    total = max((out_length-1)*stride+kernel-x.shape[1],0)
    left = total//2
    padded = np.pad(x,((0,0),(left,total-left),(0,0)))
    view = np.lib.stride_tricks.sliding_window_view(padded,kernel,axis=1)[:,::stride,:,:]
    return view.transpose(0,1,3,2), left


class CNN:
    def __init__(self, c, weights=None):
        self.layers = c['conv_layers']
        self.head = 'flatten' if c.get('input_mode','time')=='spectrum' else 'global_average'
        rng = np.random.default_rng(c['seed'])
        self.p = {}
        channels = 1
        for i,(cout,k,s) in enumerate(self.layers):
            self.p[f'w{i}'] = (rng.normal(size=(k,channels,cout))*np.sqrt(2/(k*channels))).astype(np.float32)
            self.p[f'b{i}'] = np.zeros(cout,dtype=np.float32)
            channels = cout
        length=513 if c.get('input_mode','time')=='spectrum' else 1024
        for cout,k,stride in self.layers:
            length=(length+stride-1)//stride
        dense_inputs=length*channels if self.head=='flatten' else channels
        self.p['wd'] = (rng.normal(size=(dense_inputs,len(c['classes'])))*np.sqrt(1/dense_inputs)).astype(np.float32)
        self.p['bd'] = np.zeros(len(c['classes']),dtype=np.float32)
        if weights is not None:
            if set(weights) != set(self.p):
                raise ValueError('model tensor names mismatch')
            for name, value in weights.items():
                arr = np.asarray(value,dtype=np.float32)
                if arr.shape != self.p[name].shape or not np.isfinite(arr).all():
                    raise ValueError('invalid tensor ' + name)
                self.p[name] = arr

    def forward(self, x, cache=False):
        caches = []
        for i,(_,k,s) in enumerate(self.layers):
            view,left = windows(x,k,s)
            z = view.reshape(-1,k*x.shape[2]) @ self.p[f'w{i}'].reshape(k*x.shape[2],-1)
            z = z.reshape(x.shape[0],view.shape[1],-1)+self.p[f'b{i}']
            if cache:
                caches.append((x.shape,view,left,z>0))
            x = np.maximum(z,0)
        pooled = x.reshape(x.shape[0],-1) if self.head=='flatten' else x.mean(axis=1)
        probs = softmax(pooled @ self.p['wd'] + self.p['bd'])
        return (probs,(caches,x.shape,pooled)) if cache else probs

    def gradients(self,x,y):
        probs,(caches,last_shape,pooled) = self.forward(x,True)
        loss = float(-np.log(np.maximum(probs[np.arange(len(y)),y],1e-12)).mean())
        dz = probs.copy()
        dz[np.arange(len(y)),y] -= 1
        dz /= len(y)
        grad = {'wd':pooled.T@dz,'bd':dz.sum(axis=0)}
        if self.head=='flatten':
            dx=(dz @ self.p['wd'].T).reshape(last_shape)
        else:
            dx = np.broadcast_to((dz @ self.p['wd'].T)[:,None,:]/last_shape[1],last_shape).copy()
        for i in reversed(range(len(self.layers))):
            shape,view,left,mask = caches[i]
            _,k,s = self.layers[i]
            dz = dx*mask
            flat = dz.reshape(-1,dz.shape[-1])
            grad[f'w{i}'] = (view.reshape(-1,k*shape[2]).T @ flat).reshape(self.p[f'w{i}'].shape)
            grad[f'b{i}'] = dz.sum(axis=(0,1))
            dview = (flat @ self.p[f'w{i}'].reshape(k*shape[2],-1).T).reshape(view.shape)
            padded_length = (view.shape[1]-1)*s+k
            dp = np.zeros((shape[0],padded_length,shape[2]),dtype=np.float32)
            for j in range(k):
                dp[:,j:j+s*view.shape[1]:s,:] += dview[:,:,j,:]
            dx = dp[:,left:left+shape[1],:]
        return loss, grad

    def train(self,x,y,vx,vy,c):
        rng = np.random.default_rng(c['seed'])
        m = {n:np.zeros_like(v) for n,v in self.p.items()}
        v = {n:np.zeros_like(a) for n,a in self.p.items()}
        history,best_loss,best,stale,step = [],float('inf'),None,0,0
        for epoch in range(c['epochs']):
            order = rng.permutation(len(y))
            for start in range(0,len(y),c['batch_size']):
                ids = order[start:start+c['batch_size']]
                _,g = self.gradients(x[ids],y[ids])
                step += 1
                for name in self.p:
                    if not np.isfinite(g[name]).all():
                        raise ValueError('non-finite gradient')
                    m[name] = .9*m[name]+.1*g[name]
                    v[name] = .999*v[name]+.001*g[name]**2
                    self.p[name] -= c['learning_rate']*(m[name]/(1-.9**step))/(np.sqrt(v[name]/(1-.999**step))+1e-8)
            row = {'epoch':epoch+1}
            for name,xx,yy in [('train',x,y),('validation',vx,vy)]:
                pr = self.forward(xx)
                row[name+'_loss'] = float(-np.log(np.maximum(pr[np.arange(len(yy)),yy],1e-12)).mean())
                row[name+'_accuracy'] = float((pr.argmax(1)==yy).mean())
            history.append(row)
            if row['validation_loss'] < best_loss:
                best_loss,best,stale = row['validation_loss'],{n:a.copy() for n,a in self.p.items()},0
            else:
                stale += 1
            if stale >= c['patience']:
                break
        self.p = best
        return history


def signal_features(x, peak_normalize=False):
    a = x[:,:,0].astype(np.float64)
    if peak_normalize:
        a = a/np.maximum(np.max(np.abs(a),axis=1,keepdims=True),1e-12)
    power = np.abs(np.fft.rfft(a*np.hanning(a.shape[1]),axis=1))**2
    bands = np.stack([b.sum(axis=1) for b in np.array_split(power[:,1:],16,axis=1)],axis=1)
    bands /= np.maximum(bands.sum(axis=1,keepdims=True),1e-12)
    quarters = np.stack([np.mean(q*q,axis=1) for q in np.array_split(a,4,axis=1)],axis=1)
    relative = quarters/np.maximum(quarters.sum(axis=1,keepdims=True),1e-12)
    shape = np.concatenate([bands,relative],axis=1)
    if peak_normalize:
        return shape
    return np.concatenate([shape,np.max(np.abs(a),axis=1,keepdims=True),np.sqrt(np.mean(a*a,axis=1,keepdims=True))],axis=1)


def fit_centroid(x,y):
    mean,scale = x.mean(0),np.maximum(x.std(0),1e-6)
    z = (x-mean)/scale
    centers = np.stack([z[y==i].mean(0) for i in sorted(set(y.tolist()))])
    return {'mean':mean.tolist(),'scale':scale.tolist(),'centers':centers.tolist()}


def centroid_predict(x,model):
    z = (x-np.asarray(model['mean']))/np.asarray(model['scale'])
    return (-np.sum((z[:,None,:]-np.asarray(model['centers'])[None,:,:])**2,axis=2)).argmax(1)
