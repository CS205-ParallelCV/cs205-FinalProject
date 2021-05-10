#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <unordered_map>
#include <vector>

using namespace std;

struct Vertex {
    int x; // pixel coordinate: (x,y)
    int y;
    int id;
};

struct Edge {
    int src_id;
    int dest_id;
    float weight;
};

class Graph {

private:
	// vertex id: vetex
    unordered_map<int, Vertex> vertices; 
    // adjacency list, vertex id: vector of outgoing edges
    unordered_map<int, vector<Edge> > edges; 
    int nrow; 
    int ncol;

public:
	// Constructor
    Graph(unordered_map<int, Vertex> v, unordered_map<int, vector<Edge> > e): vertices(v), edges(e) {}

    // Getter
    int num_vertices() const;
    unordered_map<int, Vertex> get_vertices() ;
    vector<int> get_vids();
    Vertex get_vertex(int v_id) ;
    vector<Edge> connected_edges(int v_id);
    bool has_edges(int v_id);
    int num_row() const;
    int num_col() const;

    // Setter 
    void set_vertices(unordered_map<int, Vertex> new_vertices);
    void set_edges(unordered_map<int, vector<Edge> > new_edges);
    void set_edges(int v_id, vector<Edge> edges);
    void set_dim(int nrow, int ncol);
    void erase_connected_edges(int id);

};

///////////////////////   GETTER   //////////////////////////
int Graph::num_vertices() const {
	return this->vertices.size();
}

// get the key of each element in the unordered map
auto key_selector = [](auto elem) {return elem.first;};

vector<int> Graph::get_vids() {
	// return a list of v_ids
	vector<int> v_ids(this->vertices.size());
	transform(this->vertices.begin(), this->vertices.end(), v_ids.begin(), key_selector);
	return v_ids;
}

unordered_map<int, Vertex> Graph::get_vertices() {
	// return a copy of unordered hashmap vertices
	unordered_map<int, Vertex> copy;
	for (auto& elem: this->vertices){
		copy[elem.first] = elem.second;
	}
	return copy;
}

Vertex Graph::get_vertex(int v_id) {
	Vertex v;
	v.x = (this->vertices)[v_id].x;
	v.y = this->vertices[v_id].y;
	v.id = this->vertices[v_id].id;
	return v;
}

vector<Edge> Graph::connected_edges(int v_id) {
	return this->edges[v_id];
}

bool Graph::has_edges(int v_id){
	return this->edges.find( v_id ) != this->edges.end() && this->edges[v_id].size()!=0;
}

int Graph::num_row() const {
	return this->nrow;
}

int Graph::num_col() const {
	return this->ncol;
}

////////////   SETTER   ////////////// 
void Graph::set_vertices(unordered_map<int, Vertex> new_vertices){
	swap(this->vertices, new_vertices);
	new_vertices.clear();
}

void Graph::set_edges(unordered_map<int, vector<Edge> > new_edges){
	swap(this->edges, new_edges);
	new_edges.clear();
}

void Graph::set_edges(int v_id, vector<Edge> edges){
	this->edges[v_id] = edges;
}

void Graph::set_dim(int nrow, int ncol){
	this->nrow = nrow;
	this->ncol = ncol;
}

void Graph::erase_connected_edges(int id){
	this->edges.erase(id);
}

