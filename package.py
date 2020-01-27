from pathlib import Path
from datetime import datetime
import subprocess
import hashlib

folder = Path(__file__).parent

# Step 1. find the sha256 of the files used for this build.
packages = [
    folder / 'graph' / "__init__.py",
    folder / 'graph' / "assignment_problem.py",
    folder / 'graph' / "flow_problem.py",
    folder / 'graph' / "graphs.py",
    folder / 'graph' / "hash.py",
    folder / 'graph' / "random.py",
    folder / 'graph' / "scheduling_problem.py",
    folder / 'graph' / "search.py",
    folder / 'graph' / "spatial_graph.py",
    folder / 'graph' / "topology.py",
    folder / 'graph' / "transform.py",
    folder / 'LICENSE',
    folder / 'README.md',
]

sha = hashlib.sha3_256()
for package_path in packages:
    assert package_path.exists()
    with open(str(package_path), mode='rb') as fi:
        data = fi.read()
        sha.update(data)

current_build_tag = sha.hexdigest()

# Step 2. get the sha256 of the existing build.
setup = folder / "setup.py"
build_tag_idx, version_idx = None, None
with open(str(setup), encoding='utf-8') as f:
    lines = f.readlines()
    for idx, row in enumerate(lines):
        if "build_tag" in row:
            if build_tag_idx is None:
                build_tag_idx = idx
            a = row.find('"') + 1
            b = row.rfind('"')
            last_build_tag = row[a:b]
        if "version=" in row and version_idx is None:
            version_idx = idx

if build_tag_idx is None:
    build_tag_idx = 0
if version_idx is None:
    raise ValueError("version not declared in setup.py")

# Step 3. compare
if current_build_tag == last_build_tag:
    print("build already in setup.py")

else:  # Step 4. make a new setup.py.
    v = datetime.now()
    version = '\"{}.{}.{}.{}\"'.format(v.year, v.month, v.day, v.hour * 3600 + v.minute * 60 + v.second)

    # update the setup.py file.
    with open(str(setup), encoding='utf-8') as fi:
        old_setup = fi.readlines()
        old_setup[build_tag_idx] = 'build_tag = "{}"\n'.format(current_build_tag)
        old_setup[version_idx] = '    version={},\n'.format(version)

    script = "".join(old_setup)
    with open(str(setup), mode='w', encoding='utf-8') as f:
        f.write(script)

    response = subprocess.Popen(["python", "setup.py", "sdist"], stdout=subprocess.PIPE)
    response.wait()
    return_code = response.returncode
    if return_code != 0:
        print(response.stdout.read().decode())
    else:
        print("new setup.py created with build_tag {}".format(current_build_tag))
        print(r"next: run: twine upload dist\graph-theory-{}.tar.gz".format(version[1:-1]))

