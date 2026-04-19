from sumo_rl import ObservationFunction
from gymnasium import spaces
import numpy as np

class RadarObservation(ObservationFunction):
    def __init__(self, ts):
        super().__init__(ts)

    def __call__(self):
        # Базовые данные: плотность машин и длина очереди
        density = self.ts.get_lanes_density()
        queue = self.ts.get_lanes_queue()
        
        # ДОБАВЛЯЕМ РАДАР: собираем среднюю скорость на каждой полосе
        speeds = []
        for lane in self.ts.lanes:
            vehicles = self.ts.sumo.lane.getLastStepVehicleIDs(lane)
            if len(vehicles) == 0:
                speeds.append(1.0) # Если полоса пуста, условная скорость максимальна
            else:
                # Высчитываем среднюю скорость потока и нормализуем ее (от 0 до 1)
                avg_speed = sum(self.ts.sumo.vehicle.getSpeed(v) for v in vehicles) / len(vehicles)
                max_speed = self.ts.sumo.lane.getMaxSpeed(lane)
                speeds.append(avg_speed / max_speed)
        
        # Склеиваем всё в один плоский массив для нейросети
        return np.array(density + queue + speeds, dtype=np.float32)

    def observation_space(self):
        # Указываем новый размер массива данных (3 параметра * количество полос)
        return spaces.Box(low=0., high=1., shape=(3 * len(self.ts.lanes),), dtype=np.float32)