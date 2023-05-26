#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the scripts directory.

# give your self permision to execute file `chmod +x ./scripts/publish_site.sh `
# run this script from the project root using `./scripts/publish_site.sh`

usage() {
  echo "Usage: $0 [-d] [-v VERSION]"
  echo ""
  echo "Build and push updated AEPsych site. Will either update latest or bump stable version."
  echo ""
  echo "  -d           Use Docusaurus bot GitHub credentials. If not specified, will use default GitHub credentials."
  echo "  -v=VERSION   Build site for new library version. If not specified, will update latest version."
  echo ""
  exit 1
}

echo "============ Building docs ==============="
./scripts/build_docs.sh -b
echo "=========================================="

# echo "===== Publishing website to gh-pages ====="
#     cd website/ || exit
#     GIT_USER=facebookresearch \
#     CURRENT_BRANCH=main \
#     USE_SSH=true \
#     yarn run publish-gh-pages
#     rm -rf build/
# echo "=========================================="


echo "======== Cleanup for github ==============="
  cd .. || exit
  find . -type d -name __pycache__ -exec rm -r {} \+
  # cd website/ || exit
  # rm -rf node_modules/
  # rm -rf yarn.lock
  # cd ../
  # cd sphinx/ || exit
  # rm -rf build/ || exit
echo "================Completed======================"
