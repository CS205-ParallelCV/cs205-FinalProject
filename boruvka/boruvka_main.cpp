#include <stdlib.h>
#include <stdio.h>
#include "opencv2/highgui.hpp"
#include <iostream>
#include <assert.h>
#include <vector>
#include <math.h> 
using namespace std;
using namespace cv;

#include "Boruvka.hpp"
#include "Graph.hpp"
#include "DisjointSet.hpp"


float weight(cv::Vec<unsigned char, 3> pixel1, cv::Vec<unsigned char, 3> pixel2 ){
    float weight = 0.0;
    for (int i=0; i < 3; i++){
        weight += pow(pixel1[i]-pixel2[i], 2.0);
    }
    return sqrt(weight);
}

Graph create_graph(Mat img){
    int nrows = img.rows;
    int ncols = img.cols;

    unordered_map<int, Vertex> vertices; 
    unordered_map<int, vector<Edge>> edges; 

    for(int i=0; i<nrows; i++){
        for(int j=0; j<ncols; j++){
            Vertex v_ij = {i, j, i*ncols + j};
            int id = i*ncols + j;
            vertices[id] = v_ij;

            vector<Edge> edges_ij;
            if( i<nrows-1 && j<ncols-1){
                // (i,j) -> (i+1,j+1)
                Edge e1;
                float w = weight(img.at<Vec3b>(i,j), img.at<Vec3b>(i+1,j+1));
                Vertex v1 = {i+1, j+1, (i+1)*ncols + j+1};
                e1.src_id = id;
                e1.dest_id = v1.id;
                e1.weight = w;

                edges_ij.push_back(e1);
            }
            if( i>0 && j>0){
                // (i,j) -> (i-1,j-1)
                Edge e2;
                float w = weight(img.at<Vec3b>(i,j), img.at<Vec3b>(i-1,j-1));
                Vertex v2 = {i-1, j-1, (i-1)*ncols + j-1};
                e2.src_id = id;
                e2.dest_id = v2.id;
                e2.weight = w;

                edges_ij.push_back(e2);
            }
            if( i>0 && j<ncols-1){
                // (i,j) -> (i-1,j+1)
                Edge e3;
                float w = weight(img.at<Vec3b>(i,j), img.at<Vec3b>(i-1,j+1));
                Vertex v3 = {i-1, j+1, (i-1)*ncols + j+1};
                e3.src_id = id;
                e3.dest_id = v3.id;
                e3.weight = w;

                edges_ij.push_back(e3);
            }
            if( i<nrows-1 && j>0){
                // (i,j) -> (i+1, j-1)
                Edge e4;
                float w = weight(img.at<Vec3b>(i,j), img.at<Vec3b>(i+1,j-1));
                Vertex v4 = {i+1, j-1, (i+1)*ncols + j-1};
                e4.src_id = id;
                e4.dest_id = v4.id;
                e4.weight = w;

                edges_ij.push_back(e4);

            }
            if( i > 0 ){
                // (i, j) -> (i-1, j)
                Edge e5;
                float w = weight(img.at<Vec3b>(i,j), img.at<Vec3b>(i-1,j));
                Vertex v5 = {i-1, j, (i-1)*ncols + j};
                e5.src_id = id;
                e5.dest_id = v5.id;
                e5.weight = w;

                edges_ij.push_back(e5);
            } 
            if( i < nrows-1){
                // (i, j) -> (i+1, j)
                Edge e6;
                float w = weight(img.at<Vec3b>(i+1,j), img.at<Vec3b>(i+1,j));
                Vertex v6 = {i+1, j, (i+1)*ncols + j};
                e6.src_id = id;
                e6.dest_id = v6.id;
                e6.weight = w;

                edges_ij.push_back(e6);
            }
            if( j > 0 ){
                // (i, j) -> (i, j-1)
                Edge e7;
                float w = weight(img.at<Vec3b>(i,j), img.at<Vec3b>(i,j-1));
                Vertex v7 = {i, j-1, i*ncols + j-1};
                e7.src_id = id;
                e7.dest_id = v7.id;
                e7.weight = w;

                edges_ij.push_back(e7);
            } 
            if( j < ncols-1){
                // (i, j) -> (i, j+1)
                Edge e8;
                float w = weight(img.at<Vec3b>(i,j), img.at<Vec3b>(i,j+1));
                Vertex v8 = {i, j+1, i*ncols + j+1};
                e8.src_id = id;
                e8.dest_id = v8.id;
                e8.weight = w;

                edges_ij.push_back(e8);
            }
 
            edges[id] = edges_ij;

        }
    }
    
    Graph g = Graph(vertices, edges);
    g.set_dim(nrows, ncols);

    return g;
}

void graph_test(Graph g){
    // cout << "Graph row "g.num_row() << " col " << g.num_col() << endl;
    unordered_map<int, Vertex> vertices = g.get_vertices();
    for (auto& elem: vertices){
        int x = elem.second.x ;
        int y = elem.second.y ;
        // cout << "Vertex " << elem.first << ": (" ;
        // cout << x << "," << y << ")\n" ;

        int nrows = g.num_row();
        int ncols = g.num_col();
        int id = x * ncols + y;

        assert(elem.first == id);
        cout << x << " " << y << " " << g.connected_edges(id).size() << "\n";
        for(Edge e:  g.connected_edges(id)){
            cout << "x" << e.dest_id/ncols  << " y" << e.dest_id%ncols << endl;
        }

        if( x==0 && y==0){
            assert(g.connected_edges(id).size()==3);
        } else if ( x==0 && y==ncols-1){
            assert(g.connected_edges(id).size()==3);
        } else if ( x==nrows-1 && y==0 ){
            assert(g.connected_edges(id).size()==3);
        } else if ( x==nrows-1 && y==ncols-1 ){
            assert(g.connected_edges(id).size()==3);
        } else if ( x==nrows-1 || x == 0 || y == 0 || y == ncols-1){
            assert(g.connected_edges(id).size()==5);
        } else {
            assert(g.connected_edges(id).size()==8);
        }

        for( Edge e: g.connected_edges(id)){
            assert(e.src_id == id);
            int diff = abs(id-e.dest_id);
            assert( diff==1 || diff==ncols || diff==ncols-1 || diff==ncols+1);
        }
    }
}

int main(int argc, char** argv){
    if(argc<2){
        cout << "Error: You should enter at least one argument-the image filename!" << endl;
        return 1;
    }

    string img_name = string(argv[1]);

    Mat img = imread(argv[1]);
    Graph g = create_graph(img);

    // testing mode
    if(argc==3 && atoi(argv[2])==1){
        graph_test(g);
    }
    
    // 1: pure MST graph problem, not related to image
    // vector<Edge> edges = find_MST(g);

    // 2: image modeled for a MST problem, without any sort of algorithmic optimization
    // vector<DisjointSet> edges = find_img_MST(g);
    
    // 3: image modeled for a MST problem, optimized, serial
    // DisjointSet* sets = new DisjointSet [ g.num_vertices() ];
    // find_img_MST_opt(sets, g);

    // 3: image modeled for a MST problem, optimized, serial
    DisjointSet* sets = new DisjointSet [ g.num_vertices() ];
    find_img_MST_opt(sets, g);

    Mat img_out(img.rows, img.cols, CV_8UC3, Scalar(0, 0, 0));

    
    for(int r=0; r<img.rows; r++){
        for(int c=0; c<img.cols; c++){
            int root = find(sets, r * img.cols + c);
            int color_r = root / img.cols;
            int color_c = root % img.cols;
            img_out.at<Vec3b>(r,c)[0] = img.at<Vec3b>(color_r, color_c)[0];
            img_out.at<Vec3b>(r,c)[1] = img.at<Vec3b>(color_r, color_c)[1];
            img_out.at<Vec3b>(r,c)[2] = img.at<Vec3b>(color_r, color_c)[2];

        }
    }
    delete[] sets;

    string delimiter = ".png";
    string token = img_name.substr(0, img_name.find(delimiter)); 
    imwrite( token+"out.png", img_out );
    
    return 0;
}