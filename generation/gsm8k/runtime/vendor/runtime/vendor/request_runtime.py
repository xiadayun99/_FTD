"""Attach the admitted direct switch to the unchanged complete-request engine.

Only canonical parameter installation/reset is specialized. Request text,
label scoring, immutable KV templates, GC, admission and live budgets inherit
the pinned serving implementation unchanged.
"""


def make_executor_class(original,tasks):
    class RequestExecutor(original):
        @property
        def uses_direct_transition(self):
            return self.policy=="canonical_patch_cached" and self.spec.get("canonical_transition_operator","node")=="direct"

        def reset(self):
            if not self.uses_direct_transition:return super().reset()
            from direct_transition import DirectSwitch
            from rounding_patch import PatchGroup
            policy=self.policy
            # The inherited reset installs the exact initial CPU reference,
            # clears the bank and resets all measured counters, outside timing.
            # Avoid constructing the legacy unmerge/merge state for edge banks.
            self.state=None;self.policy="host_copy_cached"
            try:super().reset()
            finally:self.policy=policy
            self.state=DirectSwitch(self.torch,self.params[0],self.factors,self.banks,
                "mrpc",self.contract,PatchGroup)
            self.fixed_bytes=self.payload_bytes+self.state.auxiliary_bytes
            if self.fixed_bytes>self.spec["cap_bytes"]:
                raise MemoryError("Direct grouped decoder descriptors exceed logical state cap")
            self.torch.cuda.synchronize();self.logical_max=self.fixed_bytes

        def install(self,target):
            if not self.uses_direct_transition:return super().install(target)
            if target not in tasks:raise ValueError("Unsupported direct-transition task")
            if not self.state.valid:raise RuntimeError("Interrupted direct transition; restore a canonical state")
            if self.state.task!=self.task:raise ValueError("Serving and direct parameter identities disagree")
            if target==self.task:return "resident"
            action=self.state.apply(target)
            self.addmm_calls=self.state.addmm_calls
            self.version="canonical:"+self.e.LOCAL_TARGET_HASHES[target]
            self.task=target;self.switches+=1
            return action
    return RequestExecutor
