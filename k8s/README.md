If your cluster is k8s-managed, you could host the reward oracle as a remote service and assign it a cluster IP for easier access.

```diff
# 1) Create the service:
kubectl create -f k8s/rm-service.yaml

# 2a) Start your job/pod with `k8s/serving.yaml` applied.
+     Remember to change the path to the readiness probe script.

# 2b) Inside the pod, start the remote server:
MOSEC_LOG_LEVEL=debug python -m oat.oracles.remote.server

# 3) With this being set up, start your experiment:
python -m oat.experiment.main \
    --preference_oracle remote \
    --remote_rm_url http://remote-rm \
    # other flags...
```

You could repeat step 2 to create as many instances as you want, which in turn supports running many experiments (step 3) in parallel.