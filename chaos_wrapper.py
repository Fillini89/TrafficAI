import gymnasium as gym
import random

class ChaosMonkeyWrapper(gym.Wrapper):
    def __init__(self, env, chaos_prob=0.001):
        super().__init__(env)
        # Шанс аварии на каждом шагу (0.001 = примерно 1 ДТП в час виртуального времени)
        self.chaos_prob = chaos_prob
        self.blocked_vehicle = None
        self.block_timer = 0

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Получаем прямой доступ к движку SUMO (TraCI)
        traci = self.env.unwrapped.sumo 
        
        # Логика поломки
        if self.block_timer > 0:
            self.block_timer -= 1
            if self.block_timer == 0 and self.blocked_vehicle in traci.vehicle.getIDList():
                # Время вышло - "чиним" машину и возвращаем контроль движку
                traci.vehicle.setColor(self.blocked_vehicle, (255, 255, 0, 255)) 
                traci.vehicle.setSpeed(self.blocked_vehicle, -1) 
                self.blocked_vehicle = None
        else:
            # Бросаем кубик на случайное ДТП
            if random.random() < self.chaos_prob:
                vehicles = traci.vehicle.getIDList()
                if vehicles:
                    # Выбираем жертву, красим в красный и жестко останавливаем на 12 шагов (60 секунд)
                    self.blocked_vehicle = random.choice(vehicles)
                    self.block_timer = 12 
                    traci.vehicle.setColor(self.blocked_vehicle, (255, 0, 0, 255))
                    traci.vehicle.setSpeed(self.blocked_vehicle, 0.0) 
                    
        return obs, reward, terminated, truncated, info