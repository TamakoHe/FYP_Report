import asyncio
import logging
from pymodbus.server import StartAsyncTcpServer
from pymodbus.datastore import ModbusSequentialDataBlock, ModbusServerContext

# Dynamic compatibility with the latest version
try:
    from pymodbus.datastore import ModbusDeviceContext
except ImportError:
    from pymodbus.datastore import ModbusSlaveContext as ModbusDeviceContext

# Enable logging to check connection status
logging.basicConfig()
log = logging.getLogger()
log.setLevel(logging.INFO)

async def update_robot_target_task(store):
    """
    This is a background task that calculates and updates target coordinates.
    These coordinates are written to the Modbus registers for the robot to read.
    """
    log.info("Started updating target coordinates...")
    
    while True:
        # Example coordinates for 7 joints (scaled by 1000 for integer transmission)
        # e.g., 1.57 rad -> 1570
        j1, j2, j3, j4, j5, j6, j7 = 1570, 0, 0, -1570, 0, 1570, 785
        
        # In pymodbus 3.13, write to local register address 1
        # When Franka requests network address 0, it maps to address 1 internally
        register_address = 1 
        values = [j1, j2, j3, j4, j5, j6, j7]
        
        # fx=4 means Input Registers for Modbus
        try:
            store.set_values(4, register_address, values)
        except AttributeError:
            store.setValues(4, register_address, values)
        
        # Update every 0.1 seconds
        await asyncio.sleep(0.1)

async def run_server():
    """
    Start the Modbus TCP Server to listen for the robot
    """
    # Initialize data blocks, increase size to 1000 to avoid "Illegal data address"
    store = ModbusDeviceContext(
        di=ModbusSequentialDataBlock(1, [0]*1000),       
        co=ModbusSequentialDataBlock(1, [0]*1000),       
        hr=ModbusSequentialDataBlock(1, [0]*1000),       
        ir=ModbusSequentialDataBlock(1, [0]*1000)
    )

    # Compatibility for device parameters
    try:
        context = ModbusServerContext(devices=store, single=True)
    except TypeError:
        context = ModbusServerContext(slaves=store, single=True)

    # Add background task
    asyncio.create_task(update_robot_target_task(store))

    # Wait for connection from the robot
    log.info("Modbus Server started on Mac, waiting for Franka connection (192.168.0.2:502) ...")
    await StartAsyncTcpServer(
        context=context,
        address=("0.0.0.0", 502)
    )

if __name__ == "__main__":
    try:
        asyncio.run(run_server())
    except KeyboardInterrupt:
        print("Server manually closed.")