#pragma once

#include <stdlib.h>

struct DisjointSet {
    int parent;
    int rank;
};


int find(vector<DisjointSet> s, int vertex){
    if(s[vertex].parent != vertex)
        s[vertex].parent = find(s, s[vertex].parent);
    return s[vertex].parent;
}

int find(DisjointSet* s, int vertex){
    if(s[vertex].parent != vertex)
        s[vertex].parent = find(s, s[vertex].parent);
    return s[vertex].parent;
}

// union by rank
void union_(vector<DisjointSet> s, int v1, int v2){
    int root1 = find(s, v1);
    int root2 = find(s, v2);

    int rank1 = s[root1].rank;
    int rank2 = s[root2].rank;
    
    if(rank1 < rank2){
        s[root1].parent = root2;
    } else if(rank1 > rank2){
        s[root2].parent = root1;
    } else {
        s[root1].parent = root2;
        s[root2].rank++;
    }
}

void union_(DisjointSet* s, int v1, int v2){
    int root1 = find(s, v1);
    int root2 = find(s, v2);

    int rank1 = s[root1].rank;
    int rank2 = s[root2].rank;
    
    if(rank1 < rank2){
        s[root1].parent = root2;
    } else if(rank1 > rank2){
        s[root2].parent = root1;
    } else {
        s[root1].parent = root2;
        s[root2].rank++;
    }
}

// parallel union operation
bool link_parallel(struct DisjointSet *sets, int x, int y){
    return __sync_bool_compare_and_swap(&sets[x].parent, x, y);
}

// parallel find operation
int find_parallel(struct DisjointSet *sets, int vertex){
    int parent = sets[vertex].parent;
    if(vertex != parent){
        int newp = find_parallel(sets, parent);
        __sync_bool_compare_and_swap(&sets[vertex].parent, parent, newp);
        return newp;
    }
    else{
        return vertex;
    }
}  

void union_parallel(struct DisjointSet *sets, int x, int y){
    while(!link_parallel(sets, x,y)){
        while(x != sets[x].parent || y != sets[y].parent){
            x = find_parallel(sets, x);
            y = find_parallel(sets, y);
            if(x == y){
                return;
            }
        }
    }
}