#include<vector>
using namespace std;
class Solution
{
public:
    void heapify(vector<int>& arr, int N, int i)
    {
        int largest = i;
        int l = 2 * i + 1;
        int r = 2 * i + 2;

        if (l < N && arr[l] > arr[largest])
            largest = l;

        if (r < N && arr[r] > arr[largest])
            largest = r;


        if (largest != i)
        {
            swap(arr[i], arr[largest]);
            heapify(arr, N, largest);
        }
    }

    vector<int> sortArray(vector<int> arr)
    {
        int N = arr.size();
        for (int i = N / 2 - 1; i >= 0; i--)
            heapify(arr, N, i);

        for (int i = N - 1; i > 0; i--)
        {
            swap(arr[0], arr[i]);
            heapify(arr, i, 0);
        }
        return arr;
    }

};