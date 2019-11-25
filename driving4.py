import pygame
from pygame.locals import *
import time
import math
import random
import numpy as np
import os

pygame.init()

class agente():
	def __init__(self):
		self.ALPHA = 0.2
		
		self.DECREMENTO_EPSILON = 1 / 50000000
		self.EPSILON_MIN = 0.05
		self.GAMMA = 0.98

		self.NUM_ESTADOS = 4
		self.INFO_MAX =  np.array([
			1000, #posx
			1000, #posy
			359,  #angulo
			20    #velocidad
		])
		self.INFO_MIN =  np.array([
			0, #posx
			0, #posy
			0, #angulo
			0  #velocidad
		])
		self.DIVISIONES = 20 #para discretizar
		self.INFO_ANCHO = (self.INFO_MAX - self.INFO_MIN) / self.DIVISIONES 

		self.NUM_ACCIONES = 4

		if os.path.isfile('./data/data.Q.npy'):
			self.load()
		else:
			self.epsilon = 1.0
			self.Q = np.zeros(
				(
					self.DIVISIONES + 1,
					self.DIVISIONES + 1,
					self.DIVISIONES + 1,
					self.DIVISIONES + 1, 
					self.NUM_ACCIONES
				), dtype=float
			)

	def load(self):
		self.Q = np.load('./data/data.Q.npy')
		self.epsilon = np.load('./data/data.epsilon.npy')[0]
		print("loaded")

	def save(self):
		np.save('./data/data.Q', self.Q)
		np.save('./data/data.epsilon', [self.epsilon])
		print("saved")

	def discretizar(self, info):
		return tuple(((info - self.INFO_MIN) / self.INFO_ANCHO).astype(int))

	def decidir(self, info, entrenamiento):
		info_discreta = self.discretizar(info)
		if self.epsilon > self.EPSILON_MIN:
			self.epsilon -= self.DECREMENTO_EPSILON

		if np.random.random() > self.epsilon or not entrenamiento:
			decision = np.argmax(self.Q[info_discreta])
		else:
			decision = random.randint(0,3)
		return decision

	def aprender(self,info, accion, recompensa, sig_info):
		info_discreta = self.discretizar(info)
		sig_info_discreta = self.discretizar(sig_info)
		td_target = recompensa + self.GAMMA * np.max(self.Q[sig_info_discreta])
		td_error = td_target - self.Q[info_discreta][accion]
		self.Q[info_discreta][accion] += self.ALPHA * td_error

class interprete():
	def __init__(self):
		self.juego = juego()
		self.agente = agente()
		self.entrenamiento = False
		self.contador_partidas = 0
		self.contador_pasos = 0
		self.mejor_recompensa = -1000000000
		self.recompensa_total = 0
		self.bucle()

	def info(self):
		return (
			self.juego.posx,
			self.juego.posy,
			self.get_angulo(self.juego.direccion),
			self.juego.velocidad
		)

	def get_angulo(self, angulo):
		if angulo < 0:
			while angulo < 0:
				angulo += 360
		return angulo % 360

	def recompensa(self):
		recompensa = 0
		if self.juego.posx > self.juego.limiteD:
			recompensa -= 100
		if self.juego.posx < self.juego.limiteI:
			recompensa -= 100
		if self.juego.posy < self.juego.limiteAr:
			recompensa -= 100
		if self.juego.posy > self.juego.limiteAb:
			recompensa -= 100
		if self.juego.velocidad < 1:
			recompensa -= 1
		if self.juego.velocidad > 3:
			recompensa += self.juego.velocidad
		return recompensa

	def bucle(self):
		while True:
			accion = self.agente.decidir(self.info(), self.entrenamiento)
			self.juego.keys[0], self.juego.keys[1], self.juego.keys[2], self.juego.keys[3] = [False,False,False,False]
			self.juego.keys[accion] = True
			info_anterior = self.info()
			self.juego.paso(self.entrenamiento)
			self.agente.aprender(info_anterior, accion, self.recompensa(), self.info())
			self.recompensa_total += self.recompensa()
			if not self.entrenamiento:
				time.sleep(0.015)
			self.contador_pasos+= 1
			if self.juego.vida <= 0:
				self.juego = juego()
				self.contador_partidas+= 1
				if self.recompensa_total > self.mejor_recompensa:
					self.mejor_recompensa = self.recompensa_total
				print('#{} R: {}, eps: {}, r: {}'.format(
					self.contador_partidas,
					int(self.mejor_recompensa),
					round(self.agente.epsilon, 3),
					int(self.recompensa_total)
				))
				self.contador_pasos=0
				self.recompensa_total = 0
				if self.contador_partidas % 1000 == 0:
					self.agente.save()
			for event in pygame.event.get():
				# cerrar el juego
				if event.type == pygame.QUIT:
					self.agente.save()
					time.sleep(0.015)
					pygame.quit()
					exit(0)
				#cambiar entre entrenamiento y test
				if event.type == pygame.KEYDOWN:
					self.entrenamiento = not self.entrenamiento


class juego():
	def __init__(self):
		self.player = pygame.image.load("resources/images/car.png")
		self.backgrond = pygame.image.load("resources/images/track.png")
		self.screen = pygame.display.set_mode((1000,1000))
		self.keys=[False,False,False,False]

		self.posx = 500 #1
		self.posy = 500 #2
		self.direccion = random.randint(0,360) #3
		self.velocidad = 0 #4

		self.vida = 100
		self.limiteAr= 50
		self.limiteAb= 900
		self.limiteD= 900
		self.limiteI= 50

	def pintar(self):
		pygame.display.set_caption('driving')
		self.screen.fill(0)
		playerrot = pygame.transform.rotate(self.player,self.direccion)
		self.screen.blit(self.backgrond, (0,0))
		self.screen.blit(playerrot, (self.posx,self.posy))
		pygame.display.flip()

	def mover(self):
		if self.keys[0]==True:
			pass
		if self.keys[1]==True:
			self.direccion+= 2
		if self.keys[2]==True:
			self.direccion-= 2
		if self.keys[3]==True and self.velocidad < 19.8:
			self.velocidad += 0.2
			#self.velocidad-= 0.1

		movex=math.cos(self.direccion/57.29)*self.velocidad
		movey=math.sin(self.direccion/57.29)*self.velocidad
		self.posx+=movex
		self.posy-=movey

		if self.posx < 0:
			self.posx = 20
			self.vida -= 10
		if self.posy < 0:
			self.posy = 20
			self.vida  -= 10
		if self.posx > 1000:
			self.posx = 980
			self.vida -= 10
		if self.posy > 1000:
			self.posy = 980
			self.vida -= 10
		if self.velocidad > 0.05:
			self.velocidad -= 0.05
		if self.velocidad < 1:
			self.vida -= 0.05

	def paso(self, entrenamiento):
		if not entrenamiento:
			self.pintar()
		self.mover()

interprete()