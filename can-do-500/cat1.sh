for loop in `ls -1 | grep -v cat | sort -V`
do
cat $loop >> final.md
done
