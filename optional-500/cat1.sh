for loop in `ls -1 *.md | grep -v final | grep -v can | sort -V`
do
cat $loop >> final.md
done
