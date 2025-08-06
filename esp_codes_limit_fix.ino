#include <WiFi.h>
#include <PubSubClient.h>
#include <AccelStepper.h>
#include <ESP32Servo.h>
#include <FastLED.h>

#define SERVO_PIN 15
Servo pourServo;

// Wi-Fi Credentials
const char *ssid = "Redmi K40";
const char *password = "1234567890";

// MQTT Broker Details
const char *mqtt_server = "192.168.255.207";
const int mqtt_port = 1883;
const char *mqtt_user = "";
const char *mqtt_pass = "";

// MQTT Topics
const char *limit_switch_topic = "sensor/limit";
const char *distance_topic = "sensor/distance";
const char *motor_status_topic = "motor/status";
const char *drink_selection_topic = "drink/selection";
const char *ir_confirm_topic = "sensor/ir";
const char *liquid_status_topic = "sensor/liquid";
const char *stir_response_topic = "user/stir_response";

// Stepper Motor Pins
#define DIR_PIN 12
#define STEP_PIN 14
#define ENABLE_PIN 13
// DC Motor Pins for Stirring
#define ENA 18 // ENA connected to GPIO18 (PWM-capable)
#define IN1 19 // IN1 connected to GPIO19
#define IN2 23 // IN2 connected to GPIO23
// Motor Settings
#define MOTOR_STEPS 100
#define MICROSTEPS 10
#define RPM 1000
#define ACCEL 2000

// WS2812B LED Strip Settings
#define LED_PIN 2        // GPIO pin for LED data
#define NUM_LEDS 70      // Changed to 25 LEDs
#define LED_TYPE WS2812B // LED strip type
#define COLOR_ORDER GRB  // Color order (may vary by strip)
CRGB leds[NUM_LEDS];     // LED array
#define COOLING 65       // Increased cooling for more movement
#define SPARKING 150     // More sparks for livelier effect
#define FLAME_SPEED 30
#define MAX_POWER_MILLIAMPS 500

// Distance Settings
const float MAX_ALLOWED_DISTANCE = 10.0; // 10cm in your Python code
float currentDistance = 0.0;
bool cupDetected = false;
bool drinkSelected = false;
bool spriteAvailable = true;
bool effectsEnabled = true;

// Movement Settings
const float INCHES_TO_STEPS = 1000;
const float DRINK_SPACING = 4.0; // 6 inches between drinks

// Correct drink positions (6 inches apart)
const long GRAPE_POSITION = 0;                       // Home position
const long SPRITE_POSITION = 5.0 * INCHES_TO_STEPS;  // 4 inches away
const long ORANGE_POSITION = 11.0 * INCHES_TO_STEPS; // 15 inches away
// Dispensing settings
const int DISPENSE_DURATION = 7000; // 5 seconds
const int SERVO_POUR_ANGLE = 70;    // Angle for pouring
const int SERVO_REST_ANGLE = 0;     // Rest position

// Mocktail Recipes
enum Mocktail
{
    GRAPE_FIZZ,
    ORANGE_FIZZ,
    PURPLE_SUNSET,
    ORANGE_ON_TOP,
    GRAPE_ON_TOP
};

// Motor Control
AccelStepper stepper(AccelStepper::DRIVER, STEP_PIN, DIR_PIN);
WiFiClient espClient;
PubSubClient client(espClient);

enum MotorState
{
    IDLE,
    MOVING_TO_DRINK,
    WAITING_FOR_IR,
    DISPENSING,
    MOVING_TO_NEXT,
    RETURNING,
    ASKING_TO_STIR,
    STIRRING,
    WAITING_FOR_STIR_RESPONSE // Add this new state
};
MotorState motorState = IDLE;
long homePosition = 0;
Mocktail currentDrink;
unsigned long dispensingStartTime = 0;
int currentStep = 0;

struct DispensingStep
{
    long position;
    unsigned long duration;
    String expectedSensor;
};

DispensingStep dispensingSteps[3];
int totalSteps = 0;

// Fire effect variables
bool gReverseDirection = false;
CRGBPalette16 gPal;

void setup_wifi()
{
    Serial.println("\nConnecting to WiFi...");
    WiFi.begin(ssid, password);

    while (WiFi.status() != WL_CONNECTED)
    {
        delay(500);
        Serial.print(".");
    }

    Serial.println("\nWiFi connected");
    Serial.print("IP: ");
    Serial.println(WiFi.localIP());
}

void prepareDispensingSteps(Mocktail drink)
{
    totalSteps = 0;

    switch (drink)
    {
    case GRAPE_FIZZ: // 1. Grape Fizz (Grape + Sprite)
        dispensingSteps[0] = {GRAPE_POSITION, DISPENSE_DURATION, "grape"};
        dispensingSteps[1] = {SPRITE_POSITION, DISPENSE_DURATION, "sprite"};
        totalSteps = 2;
        break;

    case ORANGE_FIZZ: // 2. Orange Fizz (Orange + Sprite)
        dispensingSteps[0] = {ORANGE_POSITION, DISPENSE_DURATION, "orange"};
        dispensingSteps[1] = {SPRITE_POSITION, DISPENSE_DURATION, "sprite"};
        totalSteps = 2;
        break;

    case PURPLE_SUNSET: // 3. Purple Sunset (Grape, Orange, Sprite)
        dispensingSteps[0] = {GRAPE_POSITION, DISPENSE_DURATION, "grape"};
        dispensingSteps[1] = {ORANGE_POSITION, DISPENSE_DURATION / 2, "orange"};
        dispensingSteps[2] = {SPRITE_POSITION, DISPENSE_DURATION / 3, "sprite"};
        totalSteps = 3;
        break;

    case ORANGE_ON_TOP: // 4. Orange on Top (Sprite first, then Orange)
        // First move to Sprite (6 inches from home)
        dispensingSteps[0] = {SPRITE_POSITION, DISPENSE_DURATION, "sprite"};
        // Then move to Orange (12 inches from home, which is 6 inches from Sprite)
        dispensingSteps[1] = {ORANGE_POSITION, DISPENSE_DURATION / 2, "orange"};
        totalSteps = 2;
        break;

    case GRAPE_ON_TOP: // 5. Grape on Top (Sprite first, then Grape)
        // First move to Sprite (6 inches from home)
        dispensingSteps[0] = {SPRITE_POSITION, DISPENSE_DURATION, "sprite"};
        // Then move back to Grape (0 inches, which is 6 inches back from Sprite)
        dispensingSteps[1] = {GRAPE_POSITION, DISPENSE_DURATION / 2, "grape"};
        totalSteps = 2;
        break;
    }

    currentStep = 0;
}

String getDrinkName(Mocktail drink)
{
    switch (drink)
    {
    case GRAPE_FIZZ:
        return "Grape Fizz";
    case ORANGE_FIZZ:
        return "Orange Fizz";
    case PURPLE_SUNSET:
        return "Purple Sunset";
    case ORANGE_ON_TOP:
        return "Orange on Top";
    case GRAPE_ON_TOP:
        return "Grape on Top";
    default:
        return "Unknown Drink";
    }
}
void setLedColor(bool danger)
{
    if (danger)
    {
        // Danger state - solid red (for limit switch or IR errors)
        fill_solid(leds, NUM_LEDS, CRGB::Red);
        FastLED.show();
        effectsEnabled = false; // Disable fire effect
    }
    else
    {
        // Safe state - enable fire effect (dark pink to baby pink)
        effectsEnabled = true;
        // Fire effect will be handled in the loop()
    }
}

void Fire2012WithPalette()
{
    static byte heat[NUM_LEDS];
    static unsigned long lastUpdate = 0;

    // Only update at FLAME_SPEED intervals for smooth animation
    if (millis() - lastUpdate < FLAME_SPEED)
        return;
    lastUpdate = millis();

    // Step 1. Cool down every cell (more variation)
    for (int i = 0; i < NUM_LEDS; i++)
    {
        heat[i] = qsub8(heat[i], random8(0, ((COOLING * 15) / NUM_LEDS) + 2));
    }

    // Step 2. Heat drifts up with more organic movement
    for (int k = NUM_LEDS - 1; k >= 3; k--)
    {
        // More natural heat transfer with varied blending
        heat[k] = (heat[k - 1] + heat[k - 2] + heat[k - 3]) / 3;
    }

    // Step 3. Random sparks with varied intensity
    if (random8() < SPARKING)
    {
        int y = random8(5);                          // More sparks at the base
        heat[y] = qadd8(heat[y], random8(150, 255)); // Brighter sparks
        // Add occasional super sparks
        if (random8() < 30)
        {
            heat[y] = 255;
        }
    }

    // Step 4. Map from heat cells to LED colors with enhanced mapping
    for (int j = 0; j < NUM_LEDS; j++)
    {
        // Add some flicker to the color mapping
        byte flicker = random8(5);
        byte colorindex = scale8(qsub8(heat[j], flicker), 250);

        // Add some sparkle to the hottest parts
        if (heat[j] > 200 && random8() < 20)
        {
            leds[j] = CRGB::White;
        }
        else
        {
            leds[j] = ColorFromPalette(gPal, colorindex);
        }

        // Add glow to the hottest LEDs
        if (heat[j] > 220)
        {
            leds[j].fadeLightBy(128 - (heat[j] - 220) * 4);
        }
    }

    // Step 5. Add some subtle "smoke" at the top
    if (random8() < 50)
    {
        for (int s = NUM_LEDS - 3; s < NUM_LEDS; s++)
        {
            leds[s] = CRGB(40, 0, 60).nscale8(random8(50, 100));
        }
    }
}
void setupDCMotor()
{
    pinMode(ENA, OUTPUT);
    pinMode(IN1, OUTPUT);
    pinMode(IN2, OUTPUT);

    // Initialize motor in stopped state
    digitalWrite(IN1, LOW);
    digitalWrite(IN2, LOW);
    analogWrite(ENA, 0);
}

void stirDrink()
{
    Serial.println("Starting stirring process...");
    client.publish(motor_status_topic, "Stirring started");

    // Set motor direction (clockwise)
    digitalWrite(IN1, HIGH);
    digitalWrite(IN2, LOW);

    // Ramp up speed smoothly (0-255)
    for (int speed = 0; speed <= 200; speed += 5)
    {
        analogWrite(ENA, speed);
        delay(50);
    }

    // Stir for fixed duration (3 seconds)
    unsigned long startTime = millis();
    while (millis() - startTime < 3000)
    {
        // Visual feedback
        fill_rainbow(leds, NUM_LEDS, (millis() / 100) % 255, 7);
        FastLED.show();
        delay(50);
    }

    // Ramp down speed smoothly
    for (int speed = 200; speed >= 0; speed -= 5)
    {
        analogWrite(ENA, speed);
        delay(50);
    }

    // Ensure complete stop
    digitalWrite(IN1, LOW);
    digitalWrite(IN2, LOW);
    analogWrite(ENA, 0);

    Serial.println("Stirring complete");
    client.publish(motor_status_topic, "Stirring complete");

    // Visual feedback
    fill_solid(leds, NUM_LEDS, CRGB::Green);
    FastLED.show();
    delay(500);

    returnToHome();
}

void returnToHome()
{
    // Stop any current movement
    stepper.stop();

    // Enable outputs if they were disabled
    stepper.enableOutputs();

    // Set the state and move to home
    motorState = RETURNING;
    stepper.moveTo(homePosition);
    client.publish(motor_status_topic, "Returning to home position");
}

void startDispensingProcess()
{
    if (totalSteps == 0 || motorState != IDLE)
        return;

    stepper.enableOutputs();
    motorState = MOVING_TO_DRINK;
    currentStep = 0;

    Serial.println("Starting dispensing process");
    stepper.moveTo(dispensingSteps[currentStep].position);
    client.publish(motor_status_topic, "Moving to first ingredient");
}

void callback(char *topic, byte *payload, unsigned int length)
{
    char message[length + 1];
    memcpy(message, payload, length);
    message[length] = '\0';

    if (String(topic) == liquid_status_topic)
    {
        String status = String(message);
        if (status == "sprite_empty")
        {
            spriteAvailable = false;
            Serial.println("Warning: Sprite container is empty!");
            client.publish(motor_status_topic, "Sprite container empty - cannot make drinks with Sprite");
            // Visual warning - flash red
            fill_solid(leds, NUM_LEDS, CRGB::Red);
            FastLED.show();
            delay(500);
            setLedColor(false); // Return to normal state
        }
        else if (status == "sprite_available")
        {
            spriteAvailable = true;
            Serial.println("Sprite container refilled");
            client.publish(motor_status_topic, "Sprite available");
            // Visual confirmation - flash green
            fill_solid(leds, NUM_LEDS, CRGB::Green);
            FastLED.show();
            delay(500);
            setLedColor(false); // Return to normal state
        }
        return;
    }
    if (String(topic) == distance_topic)
    {
        currentDistance = atof(message);
        bool newCupDetected = (currentDistance <= MAX_ALLOWED_DISTANCE);

        if (newCupDetected && !cupDetected)
        {
            Serial.println("Cup detected");
            cupDetected = true;
            delay(200);
            setLedColor(false);

            if (drinkSelected)
            {
                startDispensingProcess();
            }
        }
        else if (!newCupDetected && cupDetected)
        {
            Serial.println("Cup removed");
            cupDetected = false;
        }
        return;
    }

    if (String(topic) == drink_selection_topic)
    {
        // First try to parse as integer (old method)
        int drinkIndex = atoi(message);

        // If parsing fails (returns 0), try to match string names
        if (drinkIndex == 0 && message[0] != '0')
        {
            String drinkName = String(message);
            drinkName.toLowerCase();

            if (drinkName == "grape_fizz")
                drinkIndex = 0;
            else if (drinkName == "orange_fizz")
                drinkIndex = 1;
            else if (drinkName == "purple_sunset")
                drinkIndex = 2;
            else if (drinkName == "orange_on_top")
                drinkIndex = 3;
            else if (drinkName == "grape_on_top")
                drinkIndex = 4;
        }

        if (drinkIndex >= 0 && drinkIndex <= 4)
        {
            currentDrink = static_cast<Mocktail>(drinkIndex);

            // Rest of the drink selection handling remains the same...
            // Check if drink requires Sprite and if it's available
            if ((currentDrink == GRAPE_FIZZ || currentDrink == ORANGE_FIZZ ||
                 currentDrink == PURPLE_SUNSET || currentDrink == ORANGE_ON_TOP ||
                 currentDrink == GRAPE_ON_TOP) &&
                !spriteAvailable)
            {
                client.publish(motor_status_topic, "Cannot prepare - Sprite container is empty!");
                // Visual error - red flash
                fill_solid(leds, NUM_LEDS, CRGB::Red);
                FastLED.show();
                delay(500);
                setLedColor(false);
                return;
            }

            prepareDispensingSteps(currentDrink);
            drinkSelected = true;

            Serial.print("Selected drink: ");
            Serial.println(getDrinkName(currentDrink));

            if (cupDetected)
            {
                startDispensingProcess();
            }
            else
            {
                client.publish(motor_status_topic, "Waiting for cup detection");
            }
        }
        return;
    }

    if (String(topic) == ir_confirm_topic)
    {
        String irSensor = String(message);

        if (motorState == WAITING_FOR_IR)
        {
            // Get the expected sensor for the current step
            String expected = dispensingSteps[currentStep].expectedSensor;

            if (irSensor == expected)
            {
                Serial.print("Correct IR sensor triggered for ");
                Serial.println(expected);
                motorState = DISPENSING;
                dispensingStartTime = millis();
                pourServo.write(SERVO_POUR_ANGLE);
                client.publish(motor_status_topic, ("Dispensing " + expected).c_str());
            }
            else
            {
                Serial.print("Wrong IR sensor triggered. Expected: ");
                Serial.print(expected);
                Serial.print(", Got: ");
                Serial.println(irSensor);

                client.publish(motor_status_topic,
                               ("Incorrect position! Expected " + expected + " but got " + irSensor).c_str());

                // Visual error - red flash
                fill_solid(leds, NUM_LEDS, CRGB::Red);
                FastLED.show();
                delay(300);
                setLedColor(true); // Danger state
            }
        }
    }

    if (String(topic) == limit_switch_topic)
    {
        String status = String(message);
        if (status == "danger")
        {
            // Emergency stop procedure
            stepper.stop();
            stepper.disableOutputs();
            motorState = IDLE;
            setLedColor(true); // Set to red
            client.publish(motor_status_topic, "EMERGENCY STOP: Limit switch triggered");
            Serial.println("EMERGENCY STOP: Limit switch triggered");
            pourServo.write(SERVO_REST_ANGLE);

            // After emergency stop, prepare to return home
            delay(1000); // Short delay to ensure everything stops
            stepper.enableOutputs();
            motorState = RETURNING;
            stepper.moveTo(homePosition);
            client.publish(motor_status_topic, "Returning to home position after emergency stop");
        }
        else if (status == "safe")
        {
            client.publish(motor_status_topic, "Limit switch cleared");
            Serial.println("Limit switch cleared");
            setLedColor(false); // Return to fire effect
        }
        return;
    }

    // Modify the stir response handling
    if (String(topic) == stir_response_topic)
    {
        String response = String(message);
        response.toLowerCase();

        if (motorState == WAITING_FOR_STIR_RESPONSE)
        {
            if (response == "yes")
            {
                motorState = STIRRING;
                client.publish(motor_status_topic, "Starting stirring process...");
                stirDrink();
                // After stirring completes, return to home
                motorState = RETURNING;
                returnToHome();
            }
            else if (response == "no")
            {
                client.publish(motor_status_topic, "Skipping stirring");
                returnToHome();
            }
            else
            {
                client.publish(motor_status_topic, "Invalid response. Please answer 'yes' or 'no'");
            }
        }
    }
}

void moveToNextStep()
{
    currentStep++;

    if (currentStep < totalSteps)
    {
        motorState = MOVING_TO_DRINK;
        stepper.moveTo(dispensingSteps[currentStep].position);
        client.publish(motor_status_topic, "Moving to next ingredient");
    }
    else
    {
        // After last step, ask about stirring
        motorState = WAITING_FOR_STIR_RESPONSE;
        client.publish(motor_status_topic, "Dispensing complete. Would you like to stir your drink? Reply 'yes' or 'no'");
        Serial.println("Waiting for stir response...");
    }
}

// Update the checkMovementComplete function
void checkMovementComplete()
{
    if (stepper.distanceToGo() == 0)
    {
        switch (motorState)
        {
        case MOVING_TO_DRINK:
            motorState = WAITING_FOR_IR;
            client.publish(motor_status_topic, "Waiting for IR confirmation");
            Serial.println("Waiting for IR sensor");
            // Visual feedback - waiting (blue)

            FastLED.show();
            break;

        case DISPENSING:
            if (millis() - dispensingStartTime >= dispensingSteps[currentStep].duration)
            {
                pourServo.write(SERVO_REST_ANGLE);
                moveToNextStep();
            }
            break;

        case RETURNING:
            stepper.disableOutputs();
            motorState = IDLE;
            drinkSelected = false;
            client.publish(motor_status_topic, "Drink preparation complete - Ready for next order");
            Serial.println("Process complete - System ready");
            // Visual feedback - complete (green)
            fill_solid(leds, NUM_LEDS, CRGB::Green);
            FastLED.show();
            delay(1000);
            setLedColor(false); // Return to normal state
            break;

        default:
            break;
        }
    }
}

void reconnect()
{
    while (!client.connected())
    {
        Serial.print("Connecting to MQTT...");
        String clientId = "ESP32-Stepper-" + String(random(0xFFFF), HEX);

        if (client.connect(clientId.c_str(), mqtt_user, mqtt_pass))
        {
            Serial.println("connected");
            client.subscribe(distance_topic);
            client.subscribe(drink_selection_topic);
            client.subscribe(ir_confirm_topic);
            client.subscribe(liquid_status_topic);
            client.subscribe(limit_switch_topic);
            client.subscribe(stir_response_topic); // Add this line

            // Visual feedback - connection established (quick green flash)
            fill_solid(leds, NUM_LEDS, CRGB::Green);
            FastLED.show();
            delay(100);
            setLedColor(false);
        }
        else
        {
            Serial.print("failed, rc=");
            Serial.print(client.state());
            Serial.println(" retrying in 5s...");
            // Visual feedback - connection failed (red)
            fill_solid(leds, NUM_LEDS, CRGB::Red);
            FastLED.show();
            delay(5000);
        }
    }
}

// Enhanced Fire Violet Pink Palette
DEFINE_GRADIENT_PALETTE(fire_violet_pink_gp){
    0, 0, 0, 0,         // Black (background)
    64, 255, 0, 255,    // Violet
    128, 255, 20, 147,  // Deep Pink
    192, 255, 105, 180, // Hot Pink
    255, 255, 182, 193  // Light Pink
};

void setup()
{
    Serial.begin(115200);

    // Initialize LED strip
    FastLED.addLeds<LED_TYPE, LED_PIN, COLOR_ORDER>(leds, NUM_LEDS).setCorrection(TypicalLEDStrip);
    FastLED.setBrightness(64); // Start at 25% brightness (0-255)
    FastLED.setMaxPowerInVoltsAndMilliamps(5, MAX_POWER_MILLIAMPS);

    gPal = fire_violet_pink_gp;

    // Start with baby pink fire effect
    effectsEnabled = true;
    setLedColor(false);

    pinMode(ENABLE_PIN, OUTPUT);
    digitalWrite(ENABLE_PIN, HIGH);

    stepper.setEnablePin(ENABLE_PIN);
    stepper.setPinsInverted(false, false, true);
    stepper.setMaxSpeed(MOTOR_STEPS * MICROSTEPS * RPM / 60);
    stepper.setAcceleration(ACCEL);
    homePosition = stepper.currentPosition();

    pourServo.attach(SERVO_PIN);
    pourServo.write(SERVO_REST_ANGLE);
    // Initialize DC Motor
    setupDCMotor();
    setup_wifi();
    client.setServer(mqtt_server, mqtt_port);
    client.setCallback(callback);

    Serial.println("ESP32 ready - waiting for cup detection and drink selection");
}

void loop()
{
    if (!client.connected())
    {
        reconnect();
    }
    client.loop();

    stepper.run();

    // Only check movement complete if not waiting for stir response
    if (motorState != WAITING_FOR_STIR_RESPONSE)
    {
        checkMovementComplete();
    }

    // Run fire effect when in idle state and not in danger mode
    static unsigned long lastFireUpdate = 0;
    if (effectsEnabled)
    {
        // Default state - fire effect (dark pink to baby pink)
        if (millis() - lastFireUpdate > 30)
        {
            Fire2012WithPalette();
            FastLED.show();
            lastFireUpdate = millis();
        }
    }
    else
    {
        fill_solid(leds, NUM_LEDS, CRGB::Red);
        FastLED.show();
    }
}