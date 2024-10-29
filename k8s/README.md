If your cluster is k8s-managed, you could host the reward oracle as a remote service and assign it a cluster IP for easier access.

```bash
# 1) create the service:
kubectl create -f k8s/rm-service.yaml

# 2a) start your job/pod with `k8s/serving.yaml` applied.
# 2b) inside the pod, start the remote server:
python -m oat.oracles.remote.server

# 3) with this being set up, start your experiment:
python -m oat.experiment.main \
    --reward_oracle remote \
    --remote_rm_url http://remote-rm \
    # other flags...
```

You could repeat step 2 to create as many instances as you want, which in turn supports running many experiments (step 3) in parallel.