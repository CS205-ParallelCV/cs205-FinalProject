#pragma once

#include <stdlib.h>
#include <stdio.h>
#include <vector>
#include <memory.h>
#include <omp.h>
#include "Graph.hpp"
#include "DisjointSet.hpp"
#include "timing.h"

void find_img_MST_opt_omp(DisjointSet* components, Graph g){

    int n = g.num_vertices();
    struct Edge* min_edges = new struct Edge[n];
    float* threshold = (float*) calloc(n, sizeof(float)); // default, -1
    int* size = (int*) calloc(n, sizeof(int)); // default, 0
    bool* visited = (bool*) calloc(n, sizeof(bool)); // default, false

    int num_components = n;
    int prev_num_components = 0;
    int iter = 0;

        
    #pragma omp parallel for schedule(dynamic)
    for(int i=0; i < n; i++){
        DisjointSet dset;
        dset.parent = i;
        dset.rank = 0;
        components[i] = dset;
        threshold[i] = -1.0;
        visited[i] = false;
        size[i] = 1;
    }

    timing_t tstart, tend;
    double total_time_find = 0.0;
    double total_time_union = 0.0;

    while(num_components > n/50 && prev_num_components != num_components ){

        prev_num_components = num_components;

        get_time(&tstart);
         
        //find minimum weight edge out of each componenet
        #pragma omp for
        for(int id = 0; id < n; id++){
            if( g.has_edges(id) ){

                vector<Edge> edges = g.connected_edges(id);
                vector<Edge> new_edges; 
                int set1 = find(components, id);
                for(Edge e: edges){

                    int set2 = find(components, e.dest_id);

                    if(set1 != set2){

                        if(!visited[set1] || min_edges[set1].weight > e.weight){
                            #pragma omp critical 
                            {
                                min_edges[set1] = e; 
                            }
                            visited[set1] = true;
                        }
                        new_edges.push_back(e);
                    }
                }
                if(new_edges.size()>0){
                    g.set_edges(id, new_edges);
                } else {
                    g.erase_connected_edges(id);
                }
                
            }
        }
        get_time(&tend);
        total_time_find += timespec_diff(tstart, tend);

        // printf("find() time-sequential 1 iteration: %g\n", total_time_find);

        get_time(&tstart);
        // union based on min edge of each component
        for(int i = 0; i < n; i++){
            if(!visited[i]){
                continue;
            }
            int src = min_edges[i].src_id;
            int dest = min_edges[i].dest_id;

            int root1 = find(components, src);
            int root2 = find(components, dest);
            
            if(root1 != root2){
                
                // consider add to mst
                float w = min_edges[i].weight;

                if(threshold[root1] == -1.0) {
                    threshold[root1] = w;
                } 
                if(threshold[root2] == -1.0) {
                    threshold[root2] = w;
                } 

                // larger set should has smaller threshold to be chosen
                float thres1 = float(threshold[root1])+n/float(size[root1]);
                float thres2 = float(threshold[root2])+n/float(size[root2]);
                float thres_min = thres1 < thres2 ? thres1 : thres2;

                if(w <= thres_min){
                    num_components -= 1;
                    float max = threshold[root1] > threshold[root2] ? threshold[root1] : threshold[root2];
                    max = max > w ? max : w;

                    if(size[root1] > size[root2]){
                        size[root1] += size[root2];
                        components[root2].parent = root1;
                        threshold[root1] = max;
                    } else {
                        size[root2] += size[root1];
                        components[root1].parent = root2;
                        threshold[root2] = max;
                    }
                }
            }
        }
        
        #pragma omp parallel for schedule(dynamic)
        for(int i=0;i<n;i++){
            visited[i] = false;
        }
        get_time(&tend);
        
        total_time_union += timespec_diff(tstart, tend);

        // printf("union() time-sequential 1 iteration: %g\n", total_time_union);
        // printf("num_components %d\n", num_components);
        iter++;        
    }

    printf("total iterations: %d\n", iter);
    printf("find() time-sequential: %g\n", total_time_find);
    printf("union() time-sequential: %g\n", total_time_union);


    free(size);
    delete[] min_edges;
    free(threshold);
    free(visited);
}
