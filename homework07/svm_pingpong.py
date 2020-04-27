import numpy as np
import pickle
import games.pingpong.communication as comm
from games.pingpong.communication import (
    SceneInfo, GameInstruction, GameStatus, PlatformAction
)


def ml_loop(side: str):
    """
    The main loop of the machine learning process
   	This loop is run in a separate process, and communicates with the game process.
    Note that the game process won't wait for the ml process to generate the
    GameInstruction. It is possible that the frame of the GameInstruction
    is behind of the current frame in the game process. Try to decrease the fps
    to avoid this situation.
    """

# === Here is the execution order of the loop === #
# 1. Put the initialization code here.

    frame_position_history=[]
    status_position_history=[]
    ball_position_history=[]
    ball_speed_position_history=[]    
    platform_1P_position_history=[]   
    platform_2P_position_history=[]   
    
    # 2. Inform the game process that ml process is ready before start the loop.
    
        
    filename1 ="svm-1p.sav"
    model1=pickle.load(open(filename1, 'rb'))
    filename2 ="svm-2p.sav"
    model2=pickle.load(open(filename2, 'rb'))
    comm.ml_ready()
    # 3. Start an endless loop.
    while True:
    # 3.1. Receive the scene information sent from the game process.
        """scene_info = comm.get_scene_info()
        platform_center_x = scene_info.platform[0]
        inp_temp=np.array([scene_info.ball[0], scene_info.ball[1], scene_info.platform[0]])
        input=inp_temp[np.newaxis, :]"""
        scene_info = comm.get_scene_info()
        frame_position_history.append( scene_info.frame)
        status_position_history.append( scene_info.status)
        ball_position_history.append( scene_info.ball)
        ball_speed_position_history.append( scene_info.ball_speed)        
        platform_1P_position_history.append( scene_info.platform_1P)
        platform_2P_position_history.append( scene_info.platform_2P) 
        platform_1P_center_x = scene_info.platform_1P[0] + 0
        platform_2P_center_x = scene_info.platform_2P[0] + 0
        
        if len(ball_position_history)>1:
            #ball_going_down = 1
            vy=ball_position_history[-1][1]-ball_position_history[-2][1]
            vx=ball_position_history[-1][0]-ball_position_history[-2][0]
            #ball=np.array([vx,vy])
            inp_temp1=np.array([scene_info.ball[0],scene_info.ball[1],platform_1P_center_x,vx,vy])
            input_1P=inp_temp1[np.newaxis, :]
            inp_temp2=np.array([scene_info.ball[0],scene_info.ball[1],platform_2P_center_x,vx,vy])
            input_2P=inp_temp2[np.newaxis, :] 
            #print(len(ball_position_history))
            
            if side == "1P":           
                #print(model1.predict(input_1P))
                if(len(ball_position_history) > 1):
                    move1 = model1.predict(input_1P)
                else:
                    move1 = 0    
                #move= model.predict(input)
                    #print(move)
                if vy > 0:    
                    #print(move1)
                    if move1 == 1:
                        comm.send_instruction(scene_info.frame, PlatformAction.MOVE_RIGHT)
                    elif move1 == -1:
                        comm.send_instruction(scene_info.frame, PlatformAction.MOVE_LEFT)
                else:
                    if platform_1P_center_x >80:
                        comm.send_instruction(scene_info.frame, PlatformAction.MOVE_LEFT)
                    elif platform_1P_center_x <80: 
                        comm.send_instruction(scene_info.frame, PlatformAction.MOVE_RIGHT)  
                
            else:
                #print(model2.predict(input_2P))
                if(len(ball_position_history) > 1):
                    move2 = model2.predict(input_2P)
                else:
                    move2 = 0    
                #move= model.predict(input)
                    #print(move)
                if vy < 0:  
                    #print(move2)
                    if move2 == 1:
                        comm.send_instruction(scene_info.frame, PlatformAction.MOVE_RIGHT)
                    elif move2 == -1:
                        comm.send_instruction(scene_info.frame, PlatformAction.MOVE_LEFT)   
                else:
                    if platform_2P_center_x >80:
                        comm.send_instruction(scene_info.frame, PlatformAction.MOVE_LEFT)
                    elif platform_2P_center_x <80: 
                        comm.send_instruction(scene_info.frame, PlatformAction.MOVE_RIGHT)                
            
            if scene_info.status == GameStatus.GAME_1P_WIN or \
                scene_info.status == GameStatus.GAME_2P_WIN:
                comm.ml_ready()
                continue
    
    
    

        