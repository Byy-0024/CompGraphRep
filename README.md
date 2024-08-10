# CompactGraphRepresentations
This is the repositary for the supplementary metarials of an under reviewed VLDB paper.  
To get the dataset, please enter the folder "data" and run the "download.sh". All the zip files will be downloaded.

The code can be compiled with the Makefile.

To generate queries, please run `./test genq classOfQueries numOfQueries`, where `classOfQueries` should be selected from `v` (vertex query, used in bfs task) or `e` (edge query, used in edge existence task), and `numOfQueries` refers to the number of queries you want to generate. The generated queries will be saved at "./node_queries.txt" or "./edge_queries.txt".

To run the graph analytic task, pleas run `./test taskName graphName orderingMethod`. The `taskName` should be selected from `he` (edge existence test), `bfs`, `tc` (triangle counting) and `pr` (PageRank). The `orderingMethod` should be selected from `Greedy`, `MWPM` and `Serdyukov`. The `GraphName` should be selected from the same list as the reordering case.  
