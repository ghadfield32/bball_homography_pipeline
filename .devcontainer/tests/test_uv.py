# Test other critical packages
print("\n📦 Testing other critical packages...")

packages_to_test = [
    'numpy', 'pandas', 'matplotlib', 'scipy', 'sklearn', 
    'jupyterlab', 'seaborn', 'tqdm'
]

for package in packages_to_test:
    try:
        if package == 'sklearn':
            import sklearn
            version = sklearn.__version__
        else:
            module = __import__(package)
            version = getattr(module, '__version__', 'unknown')
        print(f"   ✅ {package}: {version}")
    except ImportError:
        print(f"   ❌ {package}: Not installed")
    except Exception as e:
        print(f"   ⚠️  {package}: Error - {e}")
