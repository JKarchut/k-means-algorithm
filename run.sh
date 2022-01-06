make
make -C ./parallel-kmeans/parallel-kmeans seq
./k-means1 -i test.txt -N 1000000 -n 10 -k 20 >> log.txt
./k-means2 -i test.txt -N 1000000 -n 10 -k 20 >> log.txt
parallel-kmeans/parallel-kmeans/seq_main -i test.txt -o -n 20 >> log.txt