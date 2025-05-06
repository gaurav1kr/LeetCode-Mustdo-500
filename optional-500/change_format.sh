#!/bin/bash

input_file="$1"
output_file="modified_${input_file}"

awk '/^\#\#[0-9]+ \*\*\*\*\[Problem Link\]https:\/\/leetcode\.com\/problems\// {
    print $0
    print "```cpp"
    print "// Your solution here"
    print "```"
    next
}1' "$input_file" > "$output_file"

echo "Modified file saved as $output_file"

