# MIT License
#
# Copyright (c) 2018-2019 Michael Fuerst
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import os
import json
import traceback

TEST_RESULTS_FILENAME = "test_results.txt"
DIVIDER = "**************************************************************"


def run_tests():
    # Runs all functions called test which are in the folder test named "test_*.py"
    # The function has the signature test(old_results: dict, new_results:dict) -> success
    old_results = {}
    new_results = {}
    if os.path.exists(TEST_RESULTS_FILENAME):
        with open(TEST_RESULTS_FILENAME, "r") as f:
            old_results = json.loads(f.read())

    positive = 0
    negative = 0
    failed = []
    tests = []
    if os.path.exists("tests"):
        tests = ["tests." + f.replace(".py", "") for f in os.listdir("tests")
                 if f.startswith("test_") and f.endswith(".py")]

    for test in tests:
        test_case = __import__(test, fromlist=["*"])
        for k in test_case.__dict__:
            if k.startswith("test_"):
                print(DIVIDER)
                print("Test: {}.{}".format(test, k))
                print(DIVIDER)
                try:
                    success = test_case.__dict__[k](old_results, new_results)
                except:
                    print(traceback.format_exc())
                    success = False
                if success:
                    print("Passed: {}.{}".format(test, k))
                    positive += 1
                else:
                    print("Failed: {}.{}".format(test, k))
                    negative += 1
                    failed.append("{}.{}".format(test, k))
                print()

    with open(TEST_RESULTS_FILENAME, "w") as f:
        f.write(json.dumps(new_results, indent=4, sort_keys=True))

    print()
    print(DIVIDER)
    print("Results")
    print(DIVIDER)
    print("Successfull Tests: {}".format(positive))
    print("Failed Tests: {}".format(negative))
    for name in failed:
        print("- {}".format(name))
    print()
    return negative == 0


def update_setup_py(major_release=False, minor_release=False):
    lines = ""
    with open("setup.py", "r") as f:
        lines = f.readlines()
    for i in range(len(lines)):
        if lines[i].startswith("__version__"):
            version = lines[i].split("=")[-1]
            version = version.split("'")[1].split(".")
            version[2] = str(int(version[2]) + 1)
            if minor_release:
                version[2] = '0'
                version[1] = str(int(version[1]) + 1)
            if major_release:
                version[2] = '0'
                version[1] = '0'
                version[0] = str(int(version[0]) + 1)
            version = ".".join(version)
            lines[i] = "__version__ = '" + version + "'\n"
    with open("setup.py", "w") as f:
        f.writelines(lines)

    return version


def commit_and_push(version):
    # Add changed setup and the test results.
    os.system("git add setup.py")
    os.system("git add " + TEST_RESULTS_FILENAME)
    os.system("git commit -m 'Version " + version + ".'")
    os.system("git push")
    # Also create tag for the version
    os.system("git tag " + version)
    os.system("git push origin --tags")


def pip_publish():
    print("TODO pip_publish")


def main(major_release=False, minor_release=False, dry_run=False):
    if run_tests():
        if dry_run:
            print("All tests successfull.")
        else:
            version = update_setup_py(major_release, minor_release)
            commit_and_push(version)
            pip_publish()
            print("Released")
    else:
        print("One or more tests failed!")


if __name__ == "__main__":
    import sys
    major = False
    minor = False
    dry_run = True
    if "major" in sys.argv:
        major = True
        dry_run = False
    if "minor" in sys.argv:
        minor = True
        dry_run = False
    if dry_run:
        print("Running a dry run, to make an actual release add 'major' or 'minor' as an argument.")
    main(major_release=major, minor_release=minor, dry_run=dry_run)
    if dry_run:
        print("Running a dry run, to make an actual release add 'major' or 'minor' as an argument.")
