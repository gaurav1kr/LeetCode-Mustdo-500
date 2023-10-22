#include<iostream>
#include<vector>
#include<unordered_map>

using namespace std;
enum Coin
{
	PENNY=1, 
	NICKLE=5, 
	DIME=10, 
	QUARTER=25
};

enum Item
{
	COKE=25,
	PEPSI=35,
	SODA=45
};

class CoinClass
{
	unordered_map<string, int> coinMap;
public:
	int GetDenomination(string coin)
	{
		return coinMap[coin];
	}
	void PutItem(string coin, int denomination)
	{
		coinMap[coin] = denomination;
	}
	CoinClass()
	{
		PutItem("penny", PENNY);
		PutItem("nickle", NICKLE);
		PutItem("dime", DIME);
		PutItem("quarter", QUARTER);
	}
};
class ItemClass
{
	unordered_map<string, int> itemMap;
public:
	int GetPrice(string item)
	{
		return itemMap[item];
	}
	void PutItem(string item, int price)
	{
		itemMap[item] = price;
	}
	ItemClass()
	{
		PutItem("coke", COKE);
		PutItem("pepsi", PEPSI);
		PutItem("soda", SODA);
	}
};


class VendingMachine
{
	virtual long selectItemAndGetPrice(ItemClass item) = 0;
	virtual void insertCoin(CoinClass coin) = 0;
	virtual vector<CoinClass> refund() = 0;
	virtual unordered_map < Item, v; khjgnb vbvbC nmk k;n nbbc   e   ctor<Coin >> collectItemAndChange() = 0;
	virtual void reset() = 0;
};

//Inventory starts
template <class T> class Inventory
{
private:
	unordered_map<T, int> inventory;
public:
	int getQuantity(T item)
	{
		int value = inventory[item];
		return (value == 0) ? 0 : value;
	}
	void add(T item)
	{
		int count = inventory[item];
		inventory[item] = count + 1;
	}
	void deduct(T item)
	{
		if (inventory.find(item) != inventory.end())
		{
			int count = inventory[item];
			inventory[item] = count - 1;
		}
	}
	bool hasItem(T item)
	{
		return (getQuantity(item) > 0);
	}
	void clear()
	{
		inventory.clear();
	}
	void put(T item, int quantity)
	{
		inventory[item] = quantity;
	}
};

//Inventory ends
class VendingMachineImpl : public VendingMachine
{
private:
	Inventory<Coin> cashInventory;
	Inventory<Item> itemInventory;
	long totalSales;
	Item currentItem;
	long currentBalance;
public:
	void intiailize()
	{
		for(int co = PENNY; co <= QUARTER; co++)
		{
			Coin coin_new = static_cast<Coin>(co);
			cashInventory.put(coin_new, 5);
		}

		for (int it = COKE; it <= SODA; it++)
		{
			Item item_new = static_cast<Item>(it);
			itemInventory.put(item_new, 5);
		}
	}
	VendingMachineImpl()
	{
		intiailize();
	}
	
	long selectItemAndGetPrice(Item item)
	{
		if (itemInventory.hasItem(item))
		{
			currentItem = item;
		}
	}
};
class VendingMachineFactory
{
	public:
		VendingMachineFactory()
		{

		}			
};