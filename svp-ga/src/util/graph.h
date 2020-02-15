#ifndef GRAPH_H
#define GRAPH_H

class Node
{
public:
    Node();
    ~Node();
private:
    unsigned int number;
};

class Graph
{
public:
    Graph();
    ~Graph();
private:
    Node* vertex;
};

class DirectedGarph : public Graph
{ 
public:
private:
};

class UndirectedGraph : public Graph
{
public:
private:
};

#endif
