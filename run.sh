make
echo "" >> log.txt
echo "--- TEST ---" >> log.txt
echo "dataset 1M vectors of 20 dimensions" >> log.txt
echo "--- GPU 1 ---" >> log.txt
echo "GPU 1"

./k-means1 -i test.txt -N 1000000 -n 10 -k 20 >> log.txt
echo "--- GPU 2 ---" >> log.txt
echo "GPU 2"
./k-means2 -i test.txt -N 1000000 -n 10 -k 20 >> log.txt