import os

def replace_keyword_in_package(package_path="./venv/Lib/site-packages/basicsr/data/degradations.py",
                               old_keyword="functional_tensor", new_keyword="functional"):
    with open(package_path, "r") as f:
        content = f.read()
    updated_content = content.replace(old_keyword, new_keyword)
    with open(package_path, "w") as f:
        f.write(updated_content)

