// Version 1: For newer mbed OS (6.x+)
#include "mbed.h"

UnbufferedSerial uart(PA_9, PA_10, 115200);
DigitalOut led(LED1);
#define Temp_Reg 0x00
#define Config_Reg 0x01
#define TLow_Reg 0x02
#define THigh_Reg 0x03
I2C TMP102(D4, D5);
const int TMP102Address = 0x90;
char ConfigRegisterTMP102[3];
char TemperatureRegister[2];
float Temperature;

void ConfigureTMP102()
{
    ConfigRegisterTMP102[0] = Config_Reg;
    ConfigRegisterTMP102[1] = 0x60; // Byte 1
    ConfigRegisterTMP102[2] = 0xA0; // Byte 2
    TMP102.write(TMP102Address, ConfigRegisterTMP102, 3);
}

void Set_Thermostat_Temp_Low()
{
    ConfigRegisterTMP102[0] = TLow_Reg;
    ConfigRegisterTMP102[1] = 0x14; // Byte 1 --20 Degrees Celsius
    ConfigRegisterTMP102[2] = 0x00; // Byte 2
    TMP102.write(TMP102Address, ConfigRegisterTMP102, 3);
}

void Set_Thermostat_Temp_High()
{
    ConfigRegisterTMP102[0] = THigh_Reg;
    ConfigRegisterTMP102[1] = 0x1A; // Byte 1
    ConfigRegisterTMP102[2] = 0x00; // Byte 2
    TMP102.write(TMP102Address, ConfigRegisterTMP102, 3);
}

void Set_Alert_Polarity()
{
    // Setting the POL bit to 1 for active-high ALERT pin
    ConfigRegisterTMP102[0] = Config_Reg;
    ConfigRegisterTMP102[1] |= 0x02;  // Set POL bit to 1 (active-high ALERT)
    TMP102.write(TMP102Address, ConfigRegisterTMP102, 3);
}

    

 

int main()
{
    int counter = 0;
    char msg[32];
    unsigned short M;
    char L;

    ConfigureTMP102();
    Set_Thermostat_Temp_Low();
    Set_Thermostat_Temp_High();
 
    ConfigRegisterTMP102[0] = Temp_Reg;
    TMP102.write(TMP102Address, ConfigRegisterTMP102, 1);
while(1){
    TMP102.read(TMP102Address, TemperatureRegister, 2);
    M = TemperatureRegister[0] << 4;
    L = TemperatureRegister[1] >> 4;
    M = M + L;

    // Convert the result to temperature in Celsius
    Temperature = 0.0625 * M;

    // Print the temperature to the console (optional)
    printf("Temperature Register Value = %u\n\r", M);
    printf("Temperature in Celsius= %.3f\n\r", Temperature);
     sprintf(msg, "DATA:%d\r\n", counter);
        uart.write(msg, strlen(msg));
        
        led = !led;
        counter=Temperature;
        
        
        ThisThread::sleep_for(1ms);
    
    }
 



    

}
/*
// Version 2: For older mbed OS (5.x)
#include "mbed.h"

RawSerial uart(PA_9, PA_10);
DigitalOut led(LED1);

int main() {
    uart.baud(115200);
    int counter = 0;
    
    while(1) {
        uart.printf("DATA:%d\r\n", counter);
        
        led = !led;
        counter++;
        
        if(counter > 999) counter = 0;
        
        wait(1.0);
    }
}
*/

/*
// Version 3: Using direct register access (always works)
#include "mbed.h"

DigitalOut tx_pin(PA_9);
DigitalOut led(LED1);

void send_char(char c) {
    // Simple bit-banging UART at 9600 baud
    int bit_time = 104; // microseconds for 9600 baud
    
    // Start bit
    tx_pin = 0;
    wait_us(bit_time);
    
    // Data bits (LSB first)
    for(int i = 0; i < 8; i++) {
        tx_pin = (c >> i) & 1;
        wait_us(bit_time);
    }
    
    // Stop bit
    tx_pin = 1;
    wait_us(bit_time);
}

void send_string(const char* str) {
    while(*str) {
        send_char(*str++);
    }
}

int main() {
    tx_pin = 1; // Idle high
    int counter = 0;
    char buffer[32];
    
    while(1) {
        sprintf(buffer, "DATA:%d\r\n", counter);
        send_string(buffer);
        
        led = !led;
        counter++;
        
        if(counter > 999) counter = 0;
        
        ThisThread::sleep_for(1000ms);
    }
}
*/
