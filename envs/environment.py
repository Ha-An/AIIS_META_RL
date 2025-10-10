import simpy
import numpy as np
from envs.config_SimPy import *  # Assuming this imports necessary configurations
from envs.log_SimPy import *  # Assuming this imports necessary logging functionalities


class Inventory:
    def __init__(self, env, item_id, holding_cost):
        # Initialize inventory object
        self.env = env
        self.item_id = item_id  # 0: product; others: WIP or material
        # Initial inventory level
        #print(ASSEMBLY_PROCESS)
        self.on_hand_inventory = I[ASSEMBLY_PROCESS][self.item_id]['INIT_LEVEL']
        # Inventory in transition (e.g., being delivered)
        self.in_transition_inventory = 0
        self.capacity_limit = INVEN_LEVEL_MAX  # Maximum capacity of the inventory
        
        # Daily inventory report template:
        '''
        Day / Inventory_Name / Inventory_Type / Inventory at the start of the day / Income_Inventory(Onhand) / Outgoing_inventory(Onhand) / Intransit_Inventory / Inventory at the end of the day
        '''
        self.daily_inven_report = [f"Day {self.env.now // 24+1}", I[ASSEMBLY_PROCESS][self.item_id]['NAME'],
                                   I[ASSEMBLY_PROCESS][self.item_id]['TYPE'], self.on_hand_inventory, 0, 0, 0, 0]
        
        """
        # for validation
        print(f"Holding_cost of {I[ASSEMBLY_PROCESS][self.item_id]['NAME']}:", holding_cost)
        """
        # Unit holding cost per hour
        self.unit_holding_cost = holding_cost / 24
        self.holding_cost_last_updated = 0.0  # Last time holding cost was updated

    def update_demand_quantity(self, daily_events):
        """
        Update the demand quantity and log the event.
        """
        daily_events.append(
            f"{present_daytime(self.env.now)}: Customer order of {I[ASSEMBLY_PROCESS][0]['NAME']}                                 : {I[ASSEMBLY_PROCESS][0]['DEMAND_QUANTITY']} units ")

    def update_inven_level(self, quantity_of_change, inven_type, daily_events):
        """
        Update the inventory level based on the quantity of change and log the event.
        """
        '''
        if I[self.item_id]["TYPE"] == "Material":
            if quantity_of_change < 0 and inven_type == "ON_HAND":
                self._update_report(quantity_of_change)
            elif inven_type == "IN_TRANSIT" and quantity_of_change > 0:
                self.change_qty = quantity_of_change
                self._update_report(quantity_of_change)
        else:
            self._update_report(quantity_of_change)
        '''

        if inven_type == "ON_HAND":
            # Update on-hand inventory
            Cost.cal_cost(self, "Holding cost")
            self.on_hand_inventory += quantity_of_change

            # Check if inventory exceeds capacity limit
            if self.on_hand_inventory > self.capacity_limit:
                daily_events.append(
                    f"{present_daytime(self.env.now)}: Due to the upper limit of the inventory, {I[ASSEMBLY_PROCESS][self.item_id]['NAME']} is wasted: {self.on_hand_inventory - self.capacity_limit}")
                self.on_hand_inventory = self.capacity_limit
            # Check if inventory goes negative
            if self.on_hand_inventory < 0:
                daily_events.append(
                    f"{present_daytime(self.env.now)}: Shortage of {I[self.item_id]['NAME']}: {self.capacity_limit - self.on_hand_inventory}")
                self.on_hand_inventory = 0

            self.holding_cost_last_updated = self.env.now

        elif inven_type == "IN_TRANSIT":
            # Update in-transition inventory
            self.in_transition_inventory += quantity_of_change
        self._update_report(quantity_of_change, inven_type)

    def _update_report(self, quantity_of_change, inven_type):
        """
        Update the daily inventory report based on the quantity of change.
        """
        if inven_type == "ON_HAND":
            if quantity_of_change > 0:
                # Income Inventory
                self.daily_inven_report[4] += quantity_of_change

            else:
                # Outgoing Invnetory
                self.daily_inven_report[5] -= quantity_of_change

        elif inven_type == "IN_TRANSIT":
            # In_Transit Inventory
            self.daily_inven_report[6] += quantity_of_change


class Supplier:
    def __init__(self, env, name, item_id):
        # Initialize supplier object
        self.env = env
        self.name = name
        self.item_id = item_id

    def deliver_to_manufacturer(self, procurement, material_qty, material_inventory, daily_events, lead_time_dict):
        """
        Deliver materials to the manufacturer after a certain lead time.
        """
        I[ASSEMBLY_PROCESS][self.item_id]["SUP_LEAD_TIME"] = SUP_LEAD_TIME_FUNC(lead_time_dict[self.item_id-1])
        lead_time = I[ASSEMBLY_PROCESS][self.item_id]["SUP_LEAD_TIME"]
        # Log the delivery event with lead time
        daily_events.append(
            f"{self.env.now}: {I[ASSEMBLY_PROCESS][self.item_id]['NAME']} will be delivered at {lead_time} days after         : {I[ASSEMBLY_PROCESS][self.item_id]['LOT_SIZE_ORDER']} units")

        # Wait for the lead time
        yield self.env.timeout(lead_time * 24)

        # Receive materials by calling the receive_materials method of procurement
        procurement.receive_materials(
            material_qty, material_inventory, daily_events)


class Procurement:
    def __init__(self, env, item_id, purchase_cost, setup_cost):
        self.env = env
        self.item_id = item_id
        self.unit_purchase_cost = purchase_cost
        '''
        #for validation
        print("Purchase_cost:",purchase_cost)
        '''
        self.unit_setup_cost = setup_cost
        self.order_qty = 0  # Initialize order quantity

    def receive_materials(self, material_qty, material_inventory, daily_events):
        """
        Process the receipt of materials and update inventory.
        """
        daily_events.append(
            f"==============={I[ASSEMBLY_PROCESS][self.item_id]['NAME']} Delivered ===============")

        # Update in-transition inventory (reduce it)
        material_inventory.update_inven_level(
            -material_qty, "IN_TRANSIT", daily_events)
        # Update on-hand inventory (increase it)
        material_inventory.update_inven_level(
            material_qty, "ON_HAND", daily_events)
        daily_events.append(
            f"{present_daytime(self.env.now)}: {I[ASSEMBLY_PROCESS][self.item_id]['NAME']} has delivered                             : {material_qty} units ")  # Record when Material provide

    def order_material(self, supplier, inventory, daily_events, lead_time_dict):
        """
        Place orders for materials to the supplier.
        """
        yield self.env.timeout(self.env.now)  # Wait for the next order cycle
        while True:
            daily_events.append(
                f"==============={I[ASSEMBLY_PROCESS][self.item_id]['NAME']}\'s Inventory ===============")

            # Set the order size based on LOT_SIZE_ORDER and reorder level
            # order_size = ORDER_QTY[self.item_id-1]
            #order_size = 2
            '''
            기존 코드
            '''
            order_size = I[ASSEMBLY_PROCESS][self.item_id]["LOT_SIZE_ORDER"] 
            # if order_size > 0 and inventory.on_hand_inventory < REORDER_LEVEL:
            if order_size > 0:
                daily_events.append(
                    f"{present_daytime(self.env.now)}: The Procurement ordered {I[ASSEMBLY_PROCESS][self.item_id]['NAME']}: {I[ASSEMBLY_PROCESS][self.item_id]['LOT_SIZE_ORDER']}  units  ")
                self.order_qty = order_size
                # Update in-transition inventory
                inventory.update_inven_level(
                    order_size, "IN_TRANSIT", daily_events)
                # Calculate and log order cost
                Cost.cal_cost(self, "Order cost")
                # Initiate the delivery process by calling deliver_to_manufacturer method of the supplier
                self.env.process(supplier.deliver_to_manufacturer(
                    self, order_size, inventory, daily_events, lead_time_dict))
            # Record in_transition_inventory
            daily_events.append(
                f"{present_daytime(self.env.now)}: {I[ASSEMBLY_PROCESS][self.item_id]['NAME']}\'s In_transition_inventory                    : {inventory.in_transition_inventory} units ")
            # Record inventory
            daily_events.append(
                f"{present_daytime(self.env.now)}: {I[ASSEMBLY_PROCESS][self.item_id]['NAME']}\'s Total_Inventory                            : {inventory.in_transition_inventory+inventory.on_hand_inventory} units  ")
            # Wait for the next order cycle
            yield self.env.timeout(I[ASSEMBLY_PROCESS][self.item_id]["MANU_ORDER_CYCLE"] * 24)
            # record order history


class Production:
    def __init__(self, env, name, process_id, production_rate, output, input_inventories, qnty_for_input_item, output_inventory, processing_cost, process_stop_cost):
        # Initialize production process
        self.env = env
        self.name = name
        self.process_id = process_id
        self.production_rate = production_rate
        self.output = output
        self.input_inventories = input_inventories
        self.qnty_for_input_item = qnty_for_input_item
        self.output_inventory = output_inventory
        self.processing_time = 24 / self.production_rate
        self.unit_processing_cost = processing_cost
        self.print_stop = True
        self.print_limit = True
        '''
        # for validation
        print(f"production_rate: {production_rate}")
        print(f"processing_cost of {P[ASSEMBLY_PROCESS][self.process_id]['PROCESS_COST']}:", processing_cost)
        '''

    def process_items(self, daily_events):
        """
        Simulate the production process.
        """
        while True:
            # Check if there's a shortage of input materials or WIPs
            shortage_check = False
            for inven, input_qnty in zip(self.input_inventories, self.qnty_for_input_item):
                if inven.on_hand_inventory < input_qnty:
                    shortage_check = True

            # Check if the output inventory is full
            inven_upper_limit_check = False
            if self.output_inventory.on_hand_inventory >= self.output_inventory.capacity_limit:
                inven_upper_limit_check = True

            if shortage_check:
                if self.print_stop:
                    daily_events.append(
                        "===============Process Phase===============")

                    daily_events.append(
                        f"{present_daytime(self.env.now)}: Stop {self.name} due to a shortage of input materials or WIPs")
                self.print_stop = False

                yield self.env.timeout(1)  # Check shortage every hour
            elif inven_upper_limit_check:
                if self.print_limit:
                    daily_events.append(
                        "===============Process Phase===============")
                    daily_events.append(
                        f"{present_daytime(self.env.now)}: Stop {self.name} due to the upper limit of the inventory. The output inventory is full")
                self.print_limit = False
                yield self.env.timeout(1)  # Check upper limit every hour
            else:
                daily_events.append(
                    "===============Process Phase===============")
                daily_events.append(
                    f"{present_daytime(self.env.now)}: Process {self.process_id} begins")

                # Consume input materials or WIPs
                for inven, input_qnty in zip(self.input_inventories, self.qnty_for_input_item):
                    inven.update_inven_level(-input_qnty,
                                             "ON_HAND", daily_events)
                # Process items (consume time)
                Cost.cal_cost(self, "Process cost")
                # Time correction
                yield self.env.timeout(self.processing_time-TIME_CORRECTION)
                daily_events.append(
                    "===============Result Phase================")
                # Cost Update Time Correction
                self.output_inventory.holding_cost_last_updated -= TIME_CORRECTION
                # Update the inventory level for the output item
                self.output_inventory.update_inven_level(
                    1, "ON_HAND", daily_events)
                # Cost Update Time Correction
                self.output_inventory.holding_cost_last_updated += TIME_CORRECTION
                daily_events.append(
                    f"{self.env.now+TIME_CORRECTION}: {self.output['NAME']} has been produced                         : 1 units")
                self.print_limit = True
                self.print_limit = True
                yield self.env.timeout(TIME_CORRECTION)  # Time correction


class Sales:
    def __init__(self, env, item_id, delivery_cost, setup_cost, shortage, due_date ):
        # Initialize sales process
        '''
        #for validation AP
        print("Product_cost")
        print(f"delivery_cost: {delivery_cost}\nsetup_cost: {setup_cost}\nshortagecost: {shortage}")
        '''
        self.env = env
        self.item_id = item_id
        self.due_date = due_date
        self.unit_delivery_cost = delivery_cost
        self.unit_setup_cost = setup_cost
        self.unit_shortage_cost = shortage
        self.delivery_item = 0
        self.num_shortages = 0

    def _deliver_to_cust(self, demand_size, product_inventory, daily_events):
        """
        Deliver products to customers and handle shortages if any.
        """
        yield self.env.timeout(I[ASSEMBLY_PROCESS][self.item_id]["DUE_DATE"] * 24-TIME_CORRECTION/2)  # Time Correction
        product_inventory.holding_cost_last_updated -= TIME_CORRECTION / \
            2  # Cost Update Time Correction
        # Check if products are available for delivery
        if product_inventory.on_hand_inventory < demand_size:
            # Calculate the shortage
            self.num_shortages = abs(
                product_inventory.on_hand_inventory - demand_size)
            # If there are some products available, deliver them first
            if product_inventory.on_hand_inventory > 0:
                self.delivery_item = product_inventory.on_hand_inventory
                daily_events.append(
                    f"{self.env.now+TIME_CORRECTION/2}: PRODUCT have been delivered to the customer       : {product_inventory.on_hand_inventory} units ")
                # Update inventory
                product_inventory.update_inven_level(
                    -product_inventory.on_hand_inventory, 'ON_HAND', daily_events)

            # Calculate and log shortage cost
            Cost.cal_cost(self, "Shortage cost")
            daily_events.append(
                f"{present_daytime(self.env.now+TIME_CORRECTION/2)}: Unable to deliver {self.num_shortages} units to the customer due to product shortage")

        else:
            # Deliver products to the customer
            self.delivery_item = demand_size
            product_inventory.update_inven_level(
                -demand_size, 'ON_HAND', daily_events)
            daily_events.append(
                f"{present_daytime(self.env.now)}: PRODUCT have been delivered to the customer : {demand_size} units ")

        # Cost Update Time Correction
        product_inventory.holding_cost_last_updated += TIME_CORRECTION
        Cost.cal_cost(self, "Delivery cost")
        yield self.env.timeout(TIME_CORRECTION/2)  # Time Correction

    def receive_demands(self, demand_qty, product_inventory, daily_events):
        """
        Receive demands from customers and initiate the delivery process.
        """
        # Update demand quantity in inventory
        product_inventory.update_demand_quantity(daily_events)
        # Initiate delivery process
        self.env.process(self._deliver_to_cust(
            demand_qty, product_inventory, daily_events))


class Customer:
    def __init__(self, env, name, item_id):
        # Initialize customer object
        self.env = env
        self.name = name
        self.item_id = item_id

    def order_product(self, sales, product_inventory, daily_events, scenario):
        """
        Place orders for products to the sales process.
        """
        yield self.env.timeout(self.env.now)  # Wait for the next order cycle
        while True:
            # Generate a random demand quantity
            I[ASSEMBLY_PROCESS][0]["DEMAND_QUANTITY"] = DEMAND_QTY_FUNC(scenario)
            demand_qty = I[ASSEMBLY_PROCESS][0]["DEMAND_QUANTITY"]
            # Receive demands and initiate delivery process
            sales.receive_demands(demand_qty, product_inventory, daily_events)
            # Wait for the next order cycle
            yield self.env.timeout(I[ASSEMBLY_PROCESS][0]["CUST_ORDER_CYCLE"] * 24)


class Cost:
    # Class for managing costs in the simulation
    def cal_cost(instance, cost_type):
        """
        Calculate and log different types of costs.
        """

        if cost_type == "Holding cost":
            # Calculate holding cost
            DAILY_COST_REPORT[cost_type] += instance.unit_holding_cost * instance.on_hand_inventory * (
                instance.env.now - instance.holding_cost_last_updated)
        elif cost_type == "Process cost":
            # Calculate processing cost
            DAILY_COST_REPORT[cost_type] += instance.unit_processing_cost
        elif cost_type == "Delivery cost":
            # Calculate delivery cost
            DAILY_COST_REPORT[cost_type] += instance.unit_delivery_cost * \
                instance.delivery_item + instance.unit_setup_cost
        elif cost_type == "Order cost":
            # Calculate order cost
            DAILY_COST_REPORT[cost_type] += instance.unit_purchase_cost * \
                instance.order_qty + instance.unit_setup_cost
        elif cost_type == "Shortage cost":
            # Calculate shortage cost
            DAILY_COST_REPORT[cost_type] += instance.unit_shortage_cost * \
                instance.num_shortages

    def update_cost_log(inventoryList):
        """
        Update the cost log at the end of each day.
        """
        cost = 0
        # Update holding cost
        for inven in inventoryList:
            DAILY_COST_REPORT['Holding cost'] += inven.unit_holding_cost * inven.on_hand_inventory * (
                inven.env.now - inven.holding_cost_last_updated)
            inven.holding_cost_last_updated = inven.env.now

        # Update daily total cost
        for key in DAILY_COST_REPORT.keys():
            cost += DAILY_COST_REPORT[key]

        return cost

    def clear_cost():
        """
        Clear the daily cost report.
        """
        # Clear daily report
        for key in DAILY_COST_REPORT.keys():
            DAILY_COST_REPORT[key] = 0


def create_env(I, P, daily_events):
    # Function to create the simulation environment and necessary objects
    simpy_env = simpy.Environment()  # Create a SimPy environment

    # Create an inventory for each item
    inventoryList = []
    
    for i in I[ASSEMBLY_PROCESS].keys():
        inventoryList.append(
            Inventory(simpy_env, i, I[ASSEMBLY_PROCESS][i]["HOLD_COST"]))

    # Create stakeholders (Customer, Suppliers)
    customer = Customer(simpy_env, "CUSTOMER", I[ASSEMBLY_PROCESS][0]["ID"])

    supplierList = []
    procurementList = []
    for i in I[ASSEMBLY_PROCESS].keys():
        if I[ASSEMBLY_PROCESS][i]["TYPE"] == 'Material':
            supplierList.append(Supplier(simpy_env, "SUPPLIER_" + str(i), i))
            procurementList.append(Procurement(
                simpy_env, I[ASSEMBLY_PROCESS][i]["ID"], I[ASSEMBLY_PROCESS][i]["PURCHASE_COST"], I[ASSEMBLY_PROCESS][i]["ORDER_COST_TO_SUP"]))

    # Create managers for manufacturing process, procurement process, and delivery process
    sales = Sales(simpy_env, customer.item_id,
                  I[ASSEMBLY_PROCESS][0]["DELIVERY_COST"], I[ASSEMBLY_PROCESS][0]["SETUP_COST_PRO"], I[ASSEMBLY_PROCESS][0]["SHORTAGE_COST_PRO"], I[ASSEMBLY_PROCESS][0]["DUE_DATE"])
    productionList = []
    for i in P[ASSEMBLY_PROCESS].keys():
        output_inventory = inventoryList[P[ASSEMBLY_PROCESS][i]["OUTPUT"]["ID"]]
        input_inventories = []
        for j in P[ASSEMBLY_PROCESS][i]["INPUT_TYPE_LIST"]:
            input_inventories.append(inventoryList[j["ID"]])
        productionList.append(Production(simpy_env, "PROCESS_" + str(i), P[ASSEMBLY_PROCESS][i]["ID"],
                                         P[ASSEMBLY_PROCESS][i]["PRODUCTION_RATE"], P[ASSEMBLY_PROCESS][i]["OUTPUT"], input_inventories, P[ASSEMBLY_PROCESS][i]["QNTY_FOR_INPUT_ITEM"], output_inventory, P[ASSEMBLY_PROCESS][i]["PROCESS_COST"], P[ASSEMBLY_PROCESS][i]["PROCESS_STOP_COST"]))

    return simpy_env, inventoryList, procurementList, productionList, sales, customer, supplierList, daily_events


# Event processes for SimPy simulation
def simpy_event_processes(simpy_env, inventoryList, procurementList, productionList, sales, customer, supplierList, daily_events, I, scenario):
    '''
    #for Validation AP
    print("="*30)
    print("Print I")
    print(I[ASSEMBLY_PROCESS])
    print("="*30)
    print("Print P")
    print(P[ASSEMBLY_PROCESS])
    '''
    '''
    # for validation scenario
    print(scenario)
    '''
    simpy_env.process(customer.order_product(
        sales, inventoryList[I[ASSEMBLY_PROCESS][0]["ID"]], daily_events, scenario["DEMAND"]))
    for production in productionList:
        simpy_env.process(production.process_items(daily_events))
    for i in range(len(supplierList)):
        simpy_env.process(procurementList[i].order_material(
            supplierList[i], inventoryList[supplierList[i].item_id], daily_events, scenario["LEADTIME"]))
    


def update_daily_report(inventoryList):
    # Update daily reports for inventory
    day_list = []
    day_dict = {}
    for inven in inventoryList:
        inven.daily_inven_report[-1] = inven.on_hand_inventory
        day_list = day_list+(inven.daily_inven_report)

        day_dict[f"On_Hand_{I[ASSEMBLY_PROCESS][inven.item_id]['NAME']}"] = inven.on_hand_inventory
        # daily_inven_report[4]: Income, inven.daily_inven_report[5]: Outgoing
        day_dict[f"Daily_Change_{I[ASSEMBLY_PROCESS][inven.item_id]['NAME']}"] = inven.daily_inven_report[4] - \
            inven.daily_inven_report[5]
        if INTRANSIT == 1:
            if I[ASSEMBLY_PROCESS][inven.item_id]["TYPE"] == "Material":
                # inven.daily_inven_report[6]: In_Transit
                day_dict[f"In_Transit_{I[ASSEMBLY_PROCESS][inven.item_id]['NAME']}"] = inven.daily_inven_report[6]
    DAILY_REPORTS.append(day_list)
    STATE_DICT.append(day_dict)
    # Reset report
    for inven in inventoryList:
        inven.daily_inven_report = [f"Day {inven.env.now//24+1}", I[ASSEMBLY_PROCESS
][inven.item_id]['NAME'], I[ASSEMBLY_PROCESS
][inven.item_id]['TYPE'],
                                    inven.on_hand_inventory, 0, 0, inven.in_transition_inventory, 0]  # inventory report


'''

def update_daily_report(inventoryList):
    # Update daily reports for inventory
    day_list = []
    for inven in inventoryList:
        inven.daily_inven_report[-1] = inven.on_hand_inventory
        day_list=day_list+(inven.daily_inven_report)
    DAILY_REPORTS.append(day_list)

    #Reset report
    for inven in inventoryList:
            if PRINT_SIM_REPORT:
                print(inven.daily_inven_report)
            inven.daily_inven_report = [f"Day {inven.env.now//24}", I[ASSEMBLY_PROCESS][inven.item_id]['NAME'], I[ASSEMBLY_PROCESS
][inven.item_id]['TYPE'],
                                        inven.on_hand_inventory, 0, 0, 0]  # inventory report
'''


def present_daytime(env_now):
    fill_length = len(str(SIM_TIME * 24))
    return str(int(env_now)).zfill(fill_length)