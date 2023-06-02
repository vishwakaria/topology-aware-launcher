# topology-aware-launcher

### Build
1. Install dependencies
```
python3 -m pip install --upgrade pip
python3 -m pip install --upgrade build
```

2. Build project. This will create an `aws_topology` whl in the dist/ directory.
```
python3 -m build
```

3. Install the pip whl. Something like:
```
 pip install dist/aws_topology-0.0.1-py3-none-any.whl --force-rein
stall
```