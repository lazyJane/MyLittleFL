```
python generate_data.py \
    --n_users 100 \
    --split dirichlet_non_iid_split\
    --n_components 3 \
    --alpha 5 \
    --s_frac 1.0 \
    --tr_frac 0.8 \
    --seed 12345  
```

```
python generate_data.py \
    --n_users 100 \
    --split dirichlet_non_iid_split\
    --n_components 6 \
    --alpha 0.5 \
    --s_frac 1.0 \
    --tr_frac 0.8 \
    --unseen_tasks_frac 0.2 \
    --seed 12345  
```

```
python generate_data.py \
    --n_users 100 \
    --split dirichlet_non_iid_split\
    --n_components 3 \
    --alpha 1.0 \
    --s_frac 1.0 \
    --tr_frac 0.8 \
    --unseen_tasks_frac 0.2 \
    --seed 12345  
```

```
python generate_data.py \
    --n_users 100 \
    --split dirichlet_non_iid_split\
    --n_components 3 \
    --alpha 0.05 \
    --s_frac 1.0 \
    --tr_frac 0.8 \
    --unseen_tasks_frac 0.2 \
    --seed 12345
```

```
python generate_data.py \
    --n_users 30 \
    --split split_iid\
    --s_frac 0.02 \
    --tr_frac 0.8 \
    --seed 12345 
```

```
python generate_data.py \
    --n_users 50 \
    --split pathological_non_iid_split\
    --s_frac 0.1 \
    --tr_frac 0.8 \
    --seed 12345 
```