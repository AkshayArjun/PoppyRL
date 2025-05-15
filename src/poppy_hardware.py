from pypot.creatures import PoppyTorso
from pypot.dynamixel import DxlIO


poppy = PoppyTorso()

poppy.l_arm_z.compliant= False


poppy.l_arm_z.goal_position = 20