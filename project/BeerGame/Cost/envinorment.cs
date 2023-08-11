using System;
using System.Collections.Generic;
using SimSharp;

namespace SimSharpExample
{
    public class Inventory
    {
        public int item_id;
        public int level;
        public double holding_cost;
        public double shortage_cost;
        public List<double> level_over_time;
        public List<double> inventory_cost_over_time;

        public Inventory(Simulation env, int item_id, double holding_cost, double shortage_cost, int initial_level)
        {
            this.item_id = item_id;
            this.level = initial_level;
            this.holding_cost = holding_cost;
            this.shortage_cost = shortage_cost;
            this.level_over_time = new List<double>();
            this.inventory_cost_over_time = new List<double>();
        }

        public void cal_inventory_cost()
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
        public Simulation env;
        public string name;
        public int item_id;

        public Provider(Simulation env, string name, int item_id)
        {
            this.env = env;
            this.name = name;
            this.item_id = item_id;
        }

        public IEnumerable<Event> deliver(int order_size, Inventory inventory)
        {
            // Lead time
            yield return env.Timeout(I[item_id]["SUP_LEAD_TIME"] * 24);
            inventory.level += order_size;
            Console.WriteLine($"{env.Now}: {name} has delivered {order_size} units of {I[item_id]['NAME']}");
        }
    }

    public class Procurement
    {
        public Simulation env;
        public int item_id;
        public double purchase_cost;
        public double setup_cost;
        public List<double> purchase_cost_over_time;
        public List<double> setup_cost_over_time;
        public double daily_procurement_cost;

        public Procurement(Simulation env, int item_id, double purchase_cost, double setup_cost)
        {
            this.env = env;
            this.item_id = item_id;
            this.purchase_cost = purchase_cost;
            this.setup_cost = setup_cost;
            this.purchase_cost_over_time = new List<double>();
            this.setup_cost_over_time = new List<double>();
            this.daily_procurement_cost = 0;
        }

        public IEnumerable<Event> order(Provider provider, Inventory inventory)
        {
            while (true)
            {
                // Place an order to a provider
                yield return env.Timeout(I[item_id]["MANU_ORDER_CYCLE"] * 24);
                // THIS WILL BE AN ACTION OF THE AGENT
                int order_size = I[item_id]["LOT_SIZE_ORDER"];
                Console.WriteLine($"{env.Now}: Placed an order for {order_size} units of {I[item_id]['NAME']}");
                yield return env.Process(provider.deliver(order_size, inventory));
                cal_procurement_cost();
            }
        }

        public void cal_procurement_cost()
        {
            daily_procurement_cost += purchase_cost * I[item_id]["LOT_SIZE_ORDER"] + setup_cost;
        }

        public void cal_daily_procurement_cost()
        {
            Console.WriteLine($"[Daily procurement cost of {I[item_id]["NAME"]}]  {daily_procurement_cost}");
            daily_procurement_cost = 0;
        }
    }

    public class Production
    {
        public Simulation env;
        public string name;
        public int process_id;
        public double production_rate;
        public Dictionary<string, object> output;
        public List<Inventory> input_inventories;
        public Inventory output_inventory;
        public double processing_cost;
        public List<double> processing_cost_over_time;
        public double daily_production_cost;

        public class Production
        {
            private readonly Simulation env;
            private readonly string name;
            private readonly int process_id;
            private readonly double production_rate;
            private readonly Dictionary<string, object> output;
            private readonly List<Inventory> input_inventories;
            private readonly Inventory output_inventory;
            private readonly double processing_cost;
            private List<double> processing_cost_over_time = new List<double>();
            private double daily_production_cost = 0;

            public Production(Simulation env, string name, int process_id, double production_rate, Dictionary<string, object> output, List<Inventory> input_inventories, Inventory output_inventory, double processing_cost)
            {
                this.env = env;
                this.name = name;
                this.process_id = process_id;
                this.production_rate = production_rate;
                this.output = output;
                this.input_inventories = input_inventories;
                this.output_inventory = output_inventory;
                this.processing_cost = processing_cost;
            }

            public IEnumerable<Event> Process()
            {
                while (true)
                {
                    // Check the current state if input materials or WIPs are available
                    bool shortage_check = false;
                    foreach (var inven in input_inventories)
                    {
                        if (inven.level < 1)
                        {
                            inven.level -= 1;
                            shortage_check = true;
                        }
                    }
                    if (shortage_check)
                    {
                        Console.WriteLine($"{env.Now}: Stop {name} due to a shortage of input materials or WIPs");
                        // Check again after 24 hours (1 day)
                        yield return env.Timeout(24);
                    }
                    else
                    {
                        // Consuming input materials or WIPs and producing output WIP or Product
                        double processing_time = 24 / production_rate;
                        yield return env.Timeout(processing_time);
                        Console.WriteLine($"{env.Now}: Process {process_id} begins");
                        foreach (var inven in input_inventories)
                        {
                            inven.level -= 1;
                            Console.WriteLine($"{env.Now}: Inventory level of {I[inven.item_id]['NAME']}: {inven.level}");
                        }
                        output_inventory.level += 1;
                        Cal_Processing_Cost(processing_time);
                        Console.WriteLine($"{env.Now}: A unit of {output['NAME']} has been produced");
                        Console.WriteLine($"{env.Now}: Inventory level of {I[output_inventory.item_id]['NAME']}: {output_inventory.level}");
                    }
                }
            }

            public void Cal_Processing_Cost(double processing_time)
            {
                daily_production_cost += processing_cost * processing_time;
            }

            public void Cal_Daily_Production_Cost()
            {
                Console.WriteLine($"[Daily production cost of {name}]  {daily_production_cost}");
                daily_production_cost = 0;
            }
        }
                   