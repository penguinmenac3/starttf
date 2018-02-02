#!/bin/bash

find data/.records ! -type d ! -name "find_unused_data.sh" -mtime +30 -atime +30 -delete
