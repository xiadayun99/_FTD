"""Target low-bit side information, shared by admitted incoming predictors.

This is raw-code lossless repair, not float quantization. General modulo/side
information coding is established; no first-invention claim is made here.
"""
from dataclasses import dataclass
import numpy as np


def pack_low2(raw):
    raw=np.asarray(raw)
    if raw.dtype!=np.uint16:raise TypeError('uint16 raw target codes required')
    codes=(raw.reshape(-1)&3).astype(np.uint8)
    if len(codes)%4:codes=np.pad(codes,(0,4-len(codes)%4))
    codes=codes.reshape(-1,4)
    return (codes[:,0]|(codes[:,1]<<2)|(codes[:,2]<<4)|(codes[:,3]<<6)).astype(np.uint8)


def unpack_low2(packed,n):
    packed=np.asarray(packed,dtype=np.uint8)
    if len(packed)!=(n+3)//4:raise ValueError('Packed target length differs')
    index=np.arange(n,dtype=np.int64)
    return ((packed[index//4]>>((index%4)*2))&3).astype(np.int32)


def encode_target(raw,escape_groups):
    raw=np.asarray(raw)
    if raw.dtype!=np.uint16:raise TypeError('uint16 raw target codes required')
    target=raw.reshape(-1);n=target.size
    if n>=2**31:raise ValueError('Projection exceeds int32 index contract')
    groups=[]
    for group in escape_groups:
        group=np.asarray(group)
        if group.ndim!=1 or group.dtype.kind not in 'iu':raise ValueError('Integer escape indices required')
        if group.size and (int(group.min())<0 or int(group.max())>=n):raise ValueError('Escape outside projection')
        groups.append(group.astype(np.int32))
    indices=np.unique(np.concatenate(groups)) if groups else np.empty(0,dtype=np.int32)
    packed=pack_low2(target)
    if 2*n<=len(packed)+6*len(indices):
        return dict(mode='dense',packed=np.empty(0,dtype=np.uint8),indices=np.empty(0,dtype=np.int32),
                    values=target.copy(),shape=raw.shape)
    return dict(mode='target_low2',packed=packed,indices=indices.astype(np.int32),values=target[indices].copy(),shape=raw.shape)


def encode_from_predictors(target,predictors):
    target=np.asarray(target,dtype=np.uint16)
    groups=[]
    for prediction in predictors:
        prediction=np.asarray(prediction,dtype=np.uint16)
        if prediction.shape!=target.shape:raise ValueError('Predictor shape differs')
        groups.append(np.flatnonzero(np.abs(target.reshape(-1).astype(np.int32)-prediction.reshape(-1).astype(np.int32))>1).astype(np.int32))
    return encode_target(target,groups)


def decode_bits(predictor,encoded):
    predictor=np.asarray(predictor)
    if predictor.dtype!=np.uint16 or tuple(predictor.shape)!=tuple(encoded['shape']):raise ValueError('Raw predictor shape/dtype differs')
    if encoded['mode']=='dense':return encoded['values'].reshape(predictor.shape).copy()
    if encoded['mode']!='target_low2':raise ValueError('Unknown target repair mode')
    raw=predictor.reshape(-1).astype(np.int32);residue=unpack_low2(encoded['packed'],raw.size)
    delta=(residue-(raw&3))&3
    step=np.where(delta==1,1,np.where(delta==3,-1,0)).astype(np.int32)
    result=((raw+step)&65535).astype(np.uint16)
    result[encoded['indices']]=encoded['values']
    return result.reshape(predictor.shape)


def payload_bytes(encoded):return sum(encoded[key].nbytes for key in ('packed','indices','values'))


@dataclass
class TargetPatch:
    shape:tuple
    dtype:str
    mode:str
    packed:object
    indices:object
    values:object
    predictor_hashes:dict
    target_sha256:str

    @property
    def nbytes(self):return sum(t.numel()*t.element_size() for t in (self.packed,self.indices,self.values))

    def to(self,device):
        return TargetPatch(self.shape,self.dtype,self.mode,self.packed.to(device),self.indices.to(device),
            self.values.to(device),dict(self.predictor_hashes),self.target_sha256)

    def metadata(self):
        return dict(shape=list(self.shape),dtype=self.dtype,mode=self.mode,bytes=self.nbytes,
            stored_escape_elements=self.indices.numel(),elements=int(np.prod(self.shape)),
            predictor_hashes=dict(self.predictor_hashes),target_sha256=self.target_sha256,
            codec='target_low2_union_v1')

    def apply_(self,tensor,source,verify_input=False):
        import torch
        import rounding_patch as rp
        if tuple(tensor.shape)!=self.shape or str(tensor.dtype)!=self.dtype or not tensor.is_contiguous():
            raise ValueError('Target patch shape/dtype/layout differs')
        if source not in self.predictor_hashes:raise ValueError('Unadmitted incoming source')
        if verify_input and rp.digest(tensor)!=self.predictor_hashes[source]:raise ValueError('Incoming predictor digest differs')
        if tensor.device!=self.packed.device:raise ValueError('Patch and parameter must be colocated')
        with torch.no_grad():
            if tensor.device.type=='cuda':
                from shared_group import SharedPatchGroup
                SharedPatchGroup({'weight':tensor},{'weight':self}).apply_()
            elif tensor.device.type=='cpu':
                encoded=dict(mode=self.mode,shape=self.shape,packed=self.packed.numpy(),indices=self.indices.numpy(),
                    values=self.values.numpy().view(np.uint16))
                raw=tensor.view(torch.int16).numpy().view(np.uint16)
                raw[:]=decode_bits(raw,encoded)
            else:raise ValueError('CPU or CUDA required')
        return tensor
