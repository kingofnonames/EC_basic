//21968
#include <bits/stdc++.h>
using namespace std;
int main(){
    freopen("./data/kp.kp", "r", stdin);
    int num_items, capacity;
    cin >> num_items >> capacity;
    vector<int> weights(num_items + 1), values(num_items + 1);
    for (int i = 1; i <= num_items; ++i){
        int w, v;
        cin >> w >> v;
        weights[i] = w;
        values[i] = v;
    }
    vector<vector<long long>> res(num_items + 1, vector<long long>(capacity + 1, 0));
    long long res_max = 0;
    for(int i = 1; i <= num_items; ++i){
        res[i][weights[i]] = values[i];
        for(int w = 0; w <= capacity; ++w){
            res[i][w] = res[i - 1][w];
            if (w - weights[i] >= 0){
                res[i][w] = max(res[i][w],  res[i - 1][w - weights[i]] + values[i]);
            }
        }
        res_max = max(res_max, res[i][capacity]);
    }
    cout << res_max << endl;
    int w = capacity;
    vector<int> chosen_items;
    for(int i = num_items; i >= 1; --i){
        if (res[i][w] != res[i-1][w]){
            chosen_items.push_back(i);
            w -= weights[i];
        }
    }
    reverse(chosen_items.begin(), chosen_items.end());
    for(int i : chosen_items) cout << i << " ";
    cout << "\n";
}