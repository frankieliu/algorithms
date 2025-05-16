# Segment Trees

## Number of nodes

A[n]
T[2*n-1]

There will be n nodes in T that match A
One level up there will be n/2, n/4 nodes etc
n/2 + n/4 + ... + 1 = ?
n(1/2 + 1/4 + 1/8 + ..1/n) = ?

tmp1     = r + r^2 + r^3 + r^4
tmp1 * r = r^2 + r^............r^5
         = tmp1 + r^5 - r
tmp1 (r-1) = r^5 - r
tmp1 = r^5 - r / (r - 1)

1/2 - 1/2 * 1/n
--------------- = 1 - 1/n
    1 - 1/2

multiplied by n = n - 1

## Problems Tutorial
Segment Tree is used in cases where there are multiple range queries on array and modifications of elements of the same array. For example, finding the sum of all the elements in an array from indices 
 to 
, or finding the minimum (famously known as Range Minumum Query problem) of all the elements in an array from indices 
 to 
. These problems can be easily solved with one of the most versatile data structures, Segment Tree.

## What is Segment Tree ?
Segment Tree is a basically a binary tree used for storing the intervals or segments. Each node in the Segment Tree represents an interval. Consider an array 
 of size 
 and a corresponding Segment Tree 
:

The root of 
 will represent the whole array 
.
Each leaf in the Segment Tree 
 will represent a single element 
 such that 
.
The internal nodes in the Segment Tree 
 represents the union of elementary intervals 
 where 
.
The root of the Segment Tree represents the whole array 
. Then it is broken down into two half intervals or segments and the two children of the root in turn represent the 
 and 
. So in each step, the segment is divided into half and the two children represent those two halves. So the height of the segment tree will be 
. There are 
 leaves representing the 
 elements of the array. The number of internal nodes is 
. So, a total number of nodes are 
.

Once the Segment Tree is built, its structure cannot be changed. We can update the values of nodes but we cannot change its structure. Segment tree provides two operations:

Update: To update the element of the array 
 and reflect the corresponding change in the Segment tree.
Query: In this operation we can query on an interval or segment and return the answer to the problem (say minimum/maximum/summation in the particular segment).
Implementation:

Since a Segment Tree is a binary tree, a simple linear array can be used to represent the Segment Tree. Before building the Segment Tree, one must figure what needs to be stored in the Segment Tree's node?.
For example, if the question is to find the sum of all the elements in an array from indices 
 to 
, then at each node (except leaf nodes) the sum of its children nodes is stored.

A Segment Tree can be built using recursion (bottom-up approach ). Start with the leaves and go up to the root and update the corresponding changes in the nodes that are in the path from leaves to root. Leaves represent a single element. In each step, the data of two children nodes are used to form an internal parent node. Each internal node will represent a union of its children’s intervals. Merging may be different for different questions. So, recursion will end up at the root node which will represent the whole array.

For 
, search the leaf that contains the element to update. This can be done by going to either on the left child or the right child depending on the interval which contains the element. Once the leaf is found, it is updated and again use the bottom-up approach to update the corresponding change in the path from that leaf to the root.

To make a 
 on the Segment Tree, select a range from 
 to 
 (which is usually given in the question). Recurse on the tree starting from the root and check if the interval represented by the node is completely in the range from 
 to 
. If the interval represented by a node is completely in the range from 
 to 
, return that node’s value.

The Segment Tree of array 
 of size 
 will look like :

enter image description here

enter image description here

Take an example. Given an array 
 of size 
 and some queries. There are two types of queries:

Update: Given 
 and 
, update array element 
 as 
.
Query: Given 
 and 
 return the value of 
 such that 
Queries and Updates can be in any order.

Naive Algorithm:
This is the most basic approach. For every query, run a loop from 
 to 
 and calculate the sum of all the elements. So each query will take 
 time.

 will update the value of the element. Each update will take 
.

This algorithm is good if the number of queries are very low compared to updates in the array.

Using Segment Tree:
First, figure what needs to be stored in the Segment Tree's node. The question asks for summation in the interval from 
 to 
, so in each node, sum of all the elements in that interval represented by the node. Next, build the Segment Tree. The implementation with comments below explains the building process.

void build(int node, int start, int end)
{
    if(start == end)
    {
        // Leaf node will have a single element
        tree[node] = A[start];
    }
    else
    {
        int mid = (start + end) / 2;
        // Recurse on the left child
        build(2*node, start, mid);
        // Recurse on the right child
        build(2*node+1, mid+1, end);
        // Internal node will have the sum of both of its children
        tree[node] = tree[2*node] + tree[2*node+1];
    }
}
As shown in the code above, start from the root and recurse on the left and the right child until a leaf node is reached. From the leaves, go back to the root and update all the nodes in the path. 
 represents the current node that is being processed. Since Segment Tree is a binary tree. 
 will represent the left node and 
 will represent the right node. 
 and 
 represents the interval represented by the node. Complexity of 
 is 
.

enter image description here

To update an element, look at the interval in which the element is present and recurse accordingly on the left or the right child.

void update(int node, int start, int end, int idx, int val)
{
    if(start == end)
    {
        // Leaf node
        A[idx] += val;
        tree[node] += val;
    }
    else
    {
        int mid = (start + end) / 2;
        if(start <= idx and idx <= mid)
        {
            // If idx is in the left child, recurse on the left child
            update(2*node, start, mid, idx, val);
        }
        else
        {
            // if idx is in the right child, recurse on the right child
            update(2*node+1, mid+1, end, idx, val);
        }
        // Internal node will have the sum of both of its children
        tree[node] = tree[2*node] + tree[2*node+1];
    }
}
Complexity of update will be 
.

To query on a given range, check 3 conditions.

Range represented by a node is completely inside the given range
Range represented by a node is completely outside the given range
Range represented by a node is partially inside and partially outside the given range
If the range represented by a node is completely outside the given range, simply return 0. If the range represented by a node is completely within the given range, return the value of the node which is the sum of all the elements in the range represented by the node. And if the range represented by a node is partially inside and partially outside the given range, return sum of the left child and the right child. Complexity of query will be 
.

int query(int node, int start, int end, int l, int r)
{
    if(r < start or end < l)
    {
        // range represented by a node is completely outside the given range
        return 0;
    }
    if(l <= start and end <= r)
    {
        // range represented by a node is completely inside the given range
        return tree[node];
    }
    // range represented by a node is partially inside and partially outside the given range
    int mid = (start + end) / 2;
    int p1 = query(2*node, start, mid, l, r);
    int p2 = query(2*node+1, mid+1, end, l, r);
    return (p1 + p2);
}
Contributed by: Akash Sharma
Did you find this tutorial helpful?
 Yes
 
 No
TEST YOUR UNDERSTANDING
Range Minimum Query
Given an array A of size N, there are two types of queries on this array.

: In this query you need to print the minimum in the sub-array 
.
: In this query you need to update 
.
Input:
First line of the test case contains two integers, N and Q, size of array A and number of queries.
Second line contains N space separated integers, elements of A.
Next Q lines contain one of the two queries.

Output:
For each type 1 query, print the minimum element in the sub-array 
.

Contraints:


SAMPLE INPUT 
5 5
1 5 2 4 3
q 1 5
q 1 3
q 3 5
u 3 6
q 1 5
SAMPLE OUTPUT 
1
1
2
1
Enter your code or Upload your code as file.
Save

C (gcc 10.3)


12345678