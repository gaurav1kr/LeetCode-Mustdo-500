file=`ls -1 /mnt/c/Users/Dell/Downloads/*.md`
cp /mnt/c/Users/Dell/Downloads/$file "$1"_"$file" 
../push_new_code.sh
rm -rf /mnt/c/Users/Dell/Downloads/*.md
