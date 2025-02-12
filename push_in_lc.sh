file=`ls -1 /mnt/c/Users/Dell/Downloads/*.md | awk -F "/" '{print $7}'`
echo "Pushing...." $file
cp /mnt/c/Users/Dell/Downloads/$file "$1"_"$file" 
../push_new_code.sh
rm -rf /mnt/c/Users/Dell/Downloads/*.md
