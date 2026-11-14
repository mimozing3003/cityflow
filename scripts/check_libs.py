for mod in ['torch', 'sklearn', 'scipy', 'numpy', 'pandas', 'matplotlib', 'seaborn', 'xgboost', 'lightgbm']:
    try:
        m = __import__(mod)
        print(f"{mod:15s}: available (v{getattr(m, '__version__', '?')})")
    except ImportError:
        print(f"{mod:15s}: NOT available")
