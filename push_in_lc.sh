file=`ls -l /mnt/c/Users/Dell/Downloads/*.md`
cp /mnt/c/Users/Dell/Downloads/$file $1_$file 
../push_new_code.sh
rm -rf /mnt/c/Users/Dell/Downloads/*.md
