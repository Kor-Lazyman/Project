using System;
using System.Collections.Generic;
using SimSharp
using System.Threading.Tasks;

public class Inventory
{
    public int item_id;
    public int level;
    public double holding_cost;
    public double shortage_cost;
    public List<double> level_over_time;
    public List<double> inventory_cost_over_time;

    public Inventory(int item_id, double holding_cost, double shortage_cost, int initial_level)
    {
        this.item_id = item_id;
        this.level = initial_level;
        this.holding_cost = holding_cost;
        this.shortage_cost = shortage_cost;
        this.level_over_time = new List<double>();
        this.inventory_cost_over_time = new List<double>();
    }

    public void cal_inventoryCost()
    {
        if (level > 0)
        {
            inventory_cost_over_time.Add(holding_cost * level);
        }
        else if (level < 0)
        {
            inventory_cost_over_time.Add(shortage_cost * Math.Abs(level));
        }
        else
        {
            inventory_cost_over_time.Add(0);
        }
        Console.WriteLine($"[Inventory Cost of {I[item_id]["NAME"]}]: {inventory_cost_over_time[inventory_cost_over_time.Count - 1]}");
    }
}

public class Provider
{
    public string name;
    public int item_id;

    public Provider(string name, int item_id)
    {
        this.name = name;
        this.item_id = item_id;
    }

    public IEnumerator<object> Deliver(int order_size, Inventory inventory)
    {
        // Lead time
        yield return env.Timeout(I[item_id]["SUP_LEAD_TIME"] * 24);
        inventory.level += order_size;
        Console.WriteLine($"{env.Now}: {name} has delivered {order_size} units of {I[item_id]['NAME']}");
    }
}

// 다른 클래스들도 동일하게 변환해야 합니다.

public class Procurement
{
    private readonly SimSharp.Environment env;
    private readonly int item_id;
    private readonly double purchase_cost;
    private readonly double setup_cost;
    private List<double> purchase_cost_over_time = new List<double>();
    private List<double> setup_cost_over_time = new List<double>();
    private double daily_procurement_cost = 0;

    public Procurement(SimSharp.Environment env, int item_id, double purchase_cost, double setup_cost)
    {
        this.env = env;
        this.item_id = item_id;
        this.purchase_cost = purchase_cost;
        this.setup_cost = setup_cost;
    }

    public async Task Order(Provider provider, Inventory inventory) //asybc:비동기 프로그래밍을 위해 사용(생략해도 상관 없으나 응답성과 안정성 측면에서 사용 하는것이 이듯)
    {
        while (true)
        {
            // Place an order to a provider
            await env.Timeout(I[item_id]["MANU_ORDER_CYCLE"] * 24);
            // THIS WILL BE AN ACTION OF THE AGENT
            int order_size = I[item_id]["LOT_SIZE_ORDER"];
            Console.WriteLine($"{env.Now}: Placed an order for {order_size} units of {I[item_id]["NAME"]}");
            await provider.Deliver(order_size, inventory);
            Cal_Procurement_cost();
        }
    }

    public void Cal_Procurement_cost()
    {
        daily_procurement_cost += purchase_cost * I[item_id]["LOT_SIZE_ORDER"] + setup_cost;
    }

    public void Cal_daily_procurementCost()
    {
        Console.WriteLine($"[Daily procurement cost of {I[item_id]["NAME"]}]  {daily_procurement_cost}");
        daily_procurement_cost = 0;
    }
} 
public static void Main(string[] args)
    {
        Console.WriteLine("디버깅 시작");
        
        // 디버깅할 코드 작성
        
        Console.WriteLine("디버깅 완료!");
    }