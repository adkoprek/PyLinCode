import sys

def hot_reload(prefixes=("/src","/tests",)):
    to_delete = []

    for name, module in list(sys.modules.items()):
        path = getattr(module, "__file__", "")
        if path:
            for prefix in prefixes:
                if path.startswith(prefix):
                    to_delete.append(name)
                    break

    for name in to_delete:
        del sys.modules[name]
