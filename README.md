# CompactGraphRepresentations
This is the repositary for the supplementary metarials of an under reviewed ICDE paper.  
To get the dataset, please enter the folder "data" and run the "download.sh". All the zip files will be downloaded.

The source code of cpu programms is in the `src` folder, and can be compiled with the Makefile. The source code of gpu programms is in the `src/gpu` folder. For each gpu graph analytic task we create a folder, which contains the makefile to compile.

To reorder the graph, please run `./test reorder graphdir orderingMethod`, where the csr format binfile of the graph dataset is stored in the folder `graphdir`. The `orderingMethod` should be selected from `Greedy`, `MWPM` and `Serdyukov`.

To generate queries, please run `./test genq classOfQueries numOfQueries`, where `classOfQueries` should be selected from `v` (vertex query, used in bfs task) or `e` (edge query, used in edge existence task), and `numOfQueries` refers to the number of queries you want to generate. The generated queries will be saved at "./node_queries.txt" or "./edge_queries.txt".

To run the graph analytic task on cpu, pleas run `./test taskName graphName orderingMethod`. The `taskName` should be selected from `bfs`, `ee` (edge existence test), `cc` (connected components), `kcore` (K-core decomposition), `pr` (PageRank), `sm` (subgraph matching) and `tc` (triangle counting). The `orderingMethod` should be selected from `origin`, `Greedy`, `MWPM` and `Serdyukov`. The `GraphName` should be selected from the same list as the reordering case.  
