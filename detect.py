#!/usr/bin/env python3
from RobotChassis import RobotChassisFun
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
from pcms.openvino_models import HumanPoseEstimation
import numpy as np
from geometry_msgs.msg import Twist
from std_msgs.msg import String
from mr_voice.msg import Voice
from PIL import Image as Images
from transformers import ViltProcessor, ViltForQuestionAnswering
import requests
import time
prev_turn =0
prev_v=0
def callback_image(msg):
    global image
    image = CvBridge().imgmsg_to_cv2(msg, "bgr8")
def callback_voice(msg):
    global voice_text, voice_direction
    voice_text = msg.text
    voice_direction = msg.direction



def callback_depth(msg):
    global depth
    depth = CvBridge().imgmsg_to_cv2(msg, "passthrough")
    
def move_linear_x(d):
    global prev_v
    if d != 0 and d <= 5000:
        v = max(min((d-1000)*0.0002, 0.4), -0.4)
        a = max(min(v-prev_v, 0.02), -0.02)
        now_v = prev_v+a
        prev_v = now_v
        return now_v
    else:
        return 0

def move_angular_z(x):
    global prev_turn
    if x>0 and x<640:
        turnv = max(min((320-x)*0.0015, 0.4), -0.4)
        turna = max(min(turnv-prev_turn, 0.1), -0.1)
        now_turnv = prev_turn + turna
        prev_turn = now_turnv
        return now_turnv
    else:
        return 0
def get_real_xyz(x, y):
    global depth
    a = 49.5 * np.pi / 180
    b = 60.0 * np.pi / 180
    d = depth[y][x]
    h, w = depth.shape[:2]
    x = x - w // 2
    y = y - h // 2
    real_y = y * 2 * d * np.tan(a / 2) / h
    real_x = x * 2 * d * np.tan(b / 2) / w
    return real_x, real_y, d
    
    
def get_pose_target(pose):
    p = []
    for i in [5, 6, 11, 12]:
        if pose[i][2] > 0:
            p.append(pose[i])
    
    if len(p) == 0: return -1, -1
    min_x = max_x = p[0][0]
    min_y = max_y = p[0][1]
    for i in range(len(p)):
        min_x = min(min_x, p[i][0])
        max_x = max(max_x, p[i][0])
        min_y = min(min_y, p[i][1])
        max_y = max(max_y, p[i][1])
    
    cx = int(min_x + max_x) // 2
    cy = int(min_y + max_y) // 2
    return cx, cy
    
    
def get_target(poses):
    target = -1
    target_d = 9999999
    for i, pose in enumerate(poses):
        cx, cy = get_pose_target(pose)
        _, _, d = get_real_xyz(cx, cy)
        if target == -1 or (d != 0 and d < target_d):
            target = i
            target_d = d
    if target == -1: return None
    return poses[target]
    
    
def calc_angular_z(cx, tx):
    e = tx - cx
    p = 0.0025
    z = p * e
    if z > 0: z = min(z, 0.3)
    if z < 0: z = max(z, -0.3)
    return z
    
    
def calc_linear_x(cd, td):
    if cd == 0: return 0
    e = cd - td
    p = 0.0005
    x = p * e
    if x > 0: x = min(x, 0.5)
    if x < 0: x = max(x, -0.5)
    return x
    
if __name__ == "__main__":
    rospy.init_node("demo2")
    rospy.loginfo("demo2 started!")
    print("1")
    # RGB Image Subscriber
    image = None
    topic_image = "/camera/rgb/image_raw"
    rospy.Subscriber(topic_image, Image, callback_image)
    rospy.wait_for_message(topic_image, Image)
    print(2)
    # Depth Image Subscriber
    depth = None
    topic_depth = "/camera/depth/image_raw"
    rospy.Subscriber(topic_depth, Image, callback_depth)
    rospy.wait_for_message(topic_depth, Image)
    speaker = rospy.Publisher("/speaker/say", String, queue_size=10, latch=True)
    print(3)
    voice_text = ""
    voice_direction = 0
    voice_topic = "/voice/text"
    rospy.Subscriber(voice_topic, Voice, callback_voice)
    print(4)
    # cmd_vel Publisher
    msg_cmd = Twist()
    pub_cmd = rospy.Publisher("/cmd_vel", Twist, queue_size=10)
    pre_x, pre_z = 0.0, 0.0
    print(5)
    # Models
    net_pose = HumanPoseEstimation()
    # Main loop
    see_if_out=0
    has_arrive=0
    speak_one_time=0
    speak_name=0
    runner=[]
    runnerindex = 0
    print("main")

    chassis = RobotChassisFun()
    print("main")
    G=0
    counter = 0
    used_name = []
    while not rospy.is_shutdown():
        print(has_arrive)
        rospy.Rate(20).sleep()
        
        time_to_go_back=0
        msg_cmd.linear.x = 0.0
        msg_cmd.angular.z = 0.0
        frame = image.copy()
        
        poses = net_pose.forward(frame)
        pose = get_target(poses)
        
        
        '''
        if pose is not None:
            cx, cy = get_pose_target(pose)
            _, _, d = get_real_xyz(cx, cy)
            cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)

            cur_z = calc_angular_z(cx, 320)
            cur_x = calc_linear_x(d, 1000)
            
            dx = cur_x - pre_x
            if dx > 0: dx = min(dx, 0.1)
            if dx < 0: dx = max(dx, -0.1)
            
            dz = cur_z - pre_z
            if dz > 0: dz = min(dz, 0.1)
            if dz < 0: dz = max(dz, -0.1)
            
            msg_cmd.linear.x = pre_x + dx
            msg_cmd.angular.z = (pre_z + dz)//2
        
        pre_x, pre_z = msg_cmd.linear.x, msg_cmd.angular.z'''
        cx=320;cy=240
        if pose is not None:
            cx, cy = get_pose_target(pose)
            cv2.circle(frame, (cx,cy), 6, (0, 255, 0), -1)
        
        cv2.circle(frame, (320,240), 5, (255, 0, 0), -1)
        if cx!=320 and cy!=240:
            msg_cmd.linear.x = move_linear_x(depth[cy][cx])
            if (900<=depth[cy][cx]<=1100 and has_arrive != 1 and see_if_out==0):
                has_arrive = 1
                see_if_out=1
        msg_cmd.angular.z = move_angular_z(cx)
        if has_arrive == 1: #Ask the person's name
            if speak_one_time==0:
                speaker.publish("What is your name?")
                
                speak_one_time += 1
            print(voice_text)
            voice_text_small = voice_text.lower()
            print(voice_text_small)
            name_list = ['tom','long','samson','yanis','andrew','james', 'john', 'robert', 'michael', 'william', 'david', 'richard', 'charles', 'joseph', 'thomas', 'christopher', 'daniel', 'paul', 'mark', 'donald', 'george', 'kenneth', 'steven', 'edward', 'brian', 'ronald', 'anthony', 'kevin', 'jason', 'matthew', 'gary', 'timothy', 'jose', 'larry', 'jeffrey', 'frank', 'scott', 'eric', 'stephen', 'andrew', 'raymond', 'gregory', 'joshua', 'jerry', 'dennis', 'walter', 'patrick', 'peter', 'harold', 'douglas', 'henry', 'carl', 'arthur', 'ryan', 'roger', 'joe', 'juan', 'jack', 'albert', 'jonathan', 'justin', 'terry', 'gerald', 'keith', 'samuel', 'willie', 'ralph', 'lawrence', 'nicholas', 'roy', 'benjamin', 'bruce', 'brandon', 'adam', 'harry', 'fred', 'wayne', 'billy', 'steve', 'louis', 'jeremy', 'aaron', 'randy', 'howard', 'eugene', 'carlos', 'russell', 'bobby', 'victor', 'martin', 'ernest', 'phillip', 'todd', 'jesse', 'craig', 'alan', 'shawn', 'clarence', 'sean', 'philip', 'chris', 'johnny', 'earl', 'jimmy', 'antonio', 'danny', 'bryan', 'tony', 'luis', 'mike', 'stanley', 'leonard', 'nathan', 'dale', 'manuel', 'rodney', 'curtis', 'norman', 'allen', 'marvin', 'vincent', 'glenn', 'jeffery', 'travis', 'jeff', 'chad', 'jacob', 'lee', 'melvin', 'alfred', 'kyle', 'francis', 'bradley', 'jesus', 'herbert', 'frederick', 'ray', 'joel', 'edwin', 'don', 'eddie', 'ricky', 'troy', 'randall', 'barry', 'alexander', 'bernard', 'mario', 'leroy', 'francisco', 'marcus', 'micheal', 'theodore', 'clifford', 'miguel', 'oscar', 'jay', 'jim', 'tom', 'calvin', 'alex', 'jon', 'ronnie', 'bill', 'lloyd', 'tommy', 'leon', 'derek', 'warren', 'darrell', 'jerome', 'floyd', 'leo', 'alvin', 'tim', 'wesley', 'gordon', 'dean', 'greg', 'jorge', 'dustin', 'pedro', 'derrick', 'dan', 'lewis', 'zachary', 'corey', 'herman', 'maurice', 'vernon', 'roberto', 'clyde', 'glen', 'hector', 'shane', 'ricardo', 'sam', 'rick', 'lester', 'brent', 'ramon', 'charlie', 'tyler', 'gilbert', 'gene', 'marc', 'reginald', 'ruben', 'brett', 'angel', 'nathaniel', 'rafael', 'leslie', 'edgar', 'milton', 'raul', 'ben', 'chester', 'cecil', 'duane', 'franklin', 'andre', 'elmer', 'brad', 'gabriel', 'ron', 'mitchell', 'roland', 'arnold', 'harvey', 'jared', 'adrian', 'karl', 'cory', 'claude', 'erik', 'darryl', 'jamie', 'neil', 'jessie', 'christian', 'javier', 'fernando', 'clinton', 'ted', 'mathew', 'tyrone', 'darren', 'lonnie', 'lance', 'cody', 'julio', 'kelly', 'kurt', 'allan', 'nelson', 'guy', 'clayton', 'hugh', 'max', 'dwayne', 'dwight', 'armando', 'felix', 'jimmie', 'everett', 'jordan', 'ian', 'wallace', 'ken', 'bob', 'jaime', 'casey', 'alfredo', 'alberto', 'dave', 'ivan', 'johnnie', 'sidney', 'byron', 'julian', 'isaac', 'morris', 'clifton', 'willard', 'daryl', 'ross', 'virgil', 'andy', 'marshall', 'salvador', 'perry', 'kirk', 'sergio', 'marion', 'tracy', 'seth', 'kent', 'terrance', 'rene', 'eduardo', 'terrence', 'enrique', 'freddie', 'wade', 'austin', 'stuart', 'fredrick', 'arturo', 'alejandro', 'jackie', 'joey', 'nick', 'luther', 'wendell', 'jeremiah', 'evan', 'julius', 'dana', 'donnie', 'otis', 'shannon', 'trevor', 'oliver', 'luke', 'homer', 'gerard', 'doug', 'kenny', 'hubert', 'angelo', 'shaun', 'lyle', 'matt', 'lynn', 'alfonso', 'orlando', 'rex', 'carlton', 'ernesto', 'cameron', 'neal', 'pablo', 'lorenzo', 'omar', 'wilbur', 'blake', 'grant', 'horace', 'roderick', 'kerry', 'abraham', 'willis', 'rickey', 'jean', 'ira', 'andres', 'cesar', 'johnathan', 'malcolm', 'rudolph', 'damon', 'kelvin', 'rudy', 'preston', 'alton', 'archie', 'marco', 'wm', 'pete', 'randolph', 'garry', 'geoffrey', 'jonathon', 'felipe', 'bennie', 'gerardo', 'ed', 'dominic', 'robin', 'loren', 'delbert', 'colin', 'guillermo', 'earnest', 'lucas', 'benny', 'noel', 'spencer', 'rodolfo', 'myron', 'edmund', 'garrett', 'salvatore', 'cedric', 'lowell', 'gregg', 'sherman', 'wilson', 'devin', 'sylvester', 'kim', 'roosevelt', 'israel', 'jermaine', 'forrest', 'wilbert', 'leland', 'simon', 'guadalupe', 'clark', 'irving', 'carroll', 'bryant', 'owen', 'rufus', 'woodrow', 'sammy', 'kristopher', 'mack', 'levi', 'marcos', 'gustavo', 'jake', 'lionel', 'marty', 'taylor', 'ellis', 'dallas', 'gilberto', 'clint', 'nicolas', 'laurence', 'ismael', 'orville', 'drew', 'jody', 'ervin', 'dewey', 'al', 'wilfred', 'josh', 'hugo', 'ignacio', 'caleb', 'tomas', 'sheldon', 'erick', 'frankie', 'stewart', 'doyle', 'darrel', 'rogelio', 'terence', 'santiago', 'alonzo', 'elias', 'bert', 'elbert', 'ramiro', 'conrad', 'pat', 'noah', 'grady', 'phil', 'cornelius', 'lamar', 'rolando', 'clay', 'percy', 'dexter', 'bradford', 'merle', 'darin', 'amos', 'terrell', 'moses', 'irvin', 'saul', 'roman', 'darnell', 'randal', 'tommie', 'timmy', 'darrin', 'winston', 'brendan', 'toby', 'van', 'abel', 'dominick', 'boyd', 'courtney', 'jan', 'emilio', 'elijah', 'cary', 'domingo', 'santos', 'aubrey', 'emmett', 'marlon', 'emanuel', 'jerald', 'edmond', 'emil', 'dewayne', 'will', 'otto', 'teddy', 'reynaldo', 'bret', 'morgan', 'jess', 'trent', 'humberto', 'emmanuel', 'stephan', 'louie', 'vicente', 'lamont', 'stacy', 'garland', 'miles', 'micah', 'efrain', 'billie', 'logan', 'heath', 'rodger', 'harley', 'demetrius', 'ethan', 'eldon', 'rocky', 'pierre', 'junior', 'freddy', 'eli', 'bryce', 'antoine', 'robbie', 'kendall', 'royce', 'sterling', 'mickey', 'chase', 'grover', 'elton', 'cleveland', 'dylan', 'chuck', 'damian', 'reuben', 'stan', 'august', 'leonardo', 'jasper', 'russel', 'erwin', 'benito', 'hans', 'monte', 'blaine', 'ernie', 'curt', 'quentin', 'agustin', 'murray', 'jamal', 'devon', 'adolfo', 'harrison', 'tyson', 'burton', 'brady', 'elliott', 'wilfredo', 'bart', 'jarrod', 'vance', 'denis', 'damien', 'joaquin', 'harlan', 'desmond', 'elliot', 'darwin', 'ashley', 'gregorio', 'buddy', 'xavier', 'kermit', 'roscoe', 'esteban', 'anton', 'solomon', 'scotty', 'norbert', 'elvin', 'williams', 'nolan', 'carey', 'rod', 'quinton', 'hal', 'brain', 'rob', 'elwood', 'kendrick', 'darius', 'moises', 'son', 'marlin', 'fidel', 'thaddeus', 'cliff', 'marcel', 'ali', 'jackson', 'raphael', 'bryon', 'armand', 'alvaro', 'jeffry', 'dane', 'joesph', 'thurman', 'ned', 'sammie', 'rusty', 'michel', 'monty', 'rory', 'fabian', 'reggie', 'mason', 'graham', 'kris', 'isaiah', 'vaughn', 'gus', 'avery', 'loyd', 'diego', 'alexis', 'adolph', 'norris', 'millard', 'rocco', 'gonzalo', 'derick', 'rodrigo', 'gerry', 'stacey', 'carmen', 'wiley', 'rigoberto', 'alphonso', 'ty', 'shelby', 'rickie', 'noe', 'vern', 'bobbie', 'reed', 'jefferson', 'elvis', 'bernardo', 'mauricio', 'hiram', 'donovan', 'basil', 'riley', 'ollie', 'nickolas', 'maynard', 'scot', 'vince', 'quincy', 'eddy', 'sebastian', 'federico', 'ulysses', 'heriberto', 'donnell', 'cole', 'denny', 'davis', 'gavin', 'emery', 'ward', 'romeo', 'jayson', 'dion', 'dante', 'clement', 'coy', 'odell', 'maxwell', 'jarvis', 'bruno', 'issac', 'mary', 'dudley', 'brock', 'sanford', 'colby', 'carmelo', 'barney', 'nestor', 'hollis', 'stefan', 'donny', 'art', 'linwood', 'beau', 'weldon', 'galen', 'isidro', 'truman', 'delmar', 'johnathon', 'silas', 'frederic', 'dick', 'kirby', 'irwin', 'cruz', 'merlin', 'merrill', 'charley', 'marcelino', 'lane', 'harris', 'cleo', 'carlo', 'trenton', 'kurtis', 'hunter', 'aurelio', 'winfred', 'vito', 'collin', 'denver', 'carter', 'leonel', 'emory', 'pasquale', 'mohammad', 'mariano', 'danial', 'blair', 'landon', 'dirk', 'branden', 'adan', 'numbers', 'clair', 'buford', 'german', 'bernie', 'wilmer', 'joan', 'emerson', 'zachery', 'fletcher', 'jacques', 'errol', 'dalton', 'monroe', 'josue', 'dominique', 'edwardo', 'booker', 'wilford', 'sonny', 'shelton', 'carson', 'theron', 'raymundo', 'daren', 'tristan', 'houston', 'robby', 'lincoln', 'jame', 'genaro', 'gale', 'bennett', 'octavio', 'cornell', 'laverne', 'hung', 'arron', 'antony', 'herschel', 'alva', 'giovanni', 'garth', 'cyrus', 'cyril', 'ronny', 'stevie', 'lon', 'freeman', 'erin', 'duncan', 'kennith', 'carmine', 'augustine', 'young', 'erich', 'chadwick', 'wilburn', 'russ', 'reid', 'myles', 'anderson', 'morton', 'jonas', 'forest', 'mitchel', 'mervin', 'zane', 'rich', 'jamel', 'lazaro', 'alphonse', 'randell', 'major', 'johnie', 'jarrett', 'brooks', 'ariel', 'abdul', 'dusty', 'luciano', 'lindsey', 'tracey', 'seymour', 'scottie', 'eugenio', 'mohammed', 'sandy', 'valentin', 'chance', 'arnulfo', 'lucien', 'ferdinand', 'thad', 'ezra', 'sydney', 'aldo', 'rubin', 'royal', 'mitch', 'earle', 'abe', 'wyatt', 'marquis', 'lanny', 'kareem', 'jamar', 'boris', 'isiah', 'emile', 'elmo', 'aron', 'leopoldo', 'everette', 'josef', 'gail', 'eloy', 'dorian', 'rodrick', 'reinaldo', 'lucio', 'jerrod', 'weston', 'hershel', 'barton', 'parker', 'lemuel', 'lavern', 'burt', 'jules', 'gil', 'eliseo', 'ahmad', 'nigel', 'efren', 'antwan', 'alden', 'margarito', 'coleman', 'refugio', 'dino', 'osvaldo', 'les', 'deandre', 'normand', 'kieth', 'ivory', 'andrea', 'trey', 'norberto', 'napoleon', 'jerold', 'fritz', 'rosendo', 'milford', 'sang', 'deon', 'christoper', 'alfonzo', 'lyman', 'josiah', 'brant', 'wilton', 'rico', 'jamaal', 'dewitt', 'carol', 'brenton', 'yong', 'olin', 'foster', 'faustino', 'claudio', 'judson', 'gino', 'edgardo', 'berry', 'alec', 'tanner', 'jarred', 'donn', 'trinidad', 'tad', 'shirley', 'prince', 'porfirio', 'odis', 'maria', 'lenard', 'chauncey', 'chang', 'tod', 'mel', 'marcelo', 'kory', 'augustus', 'keven', 'hilario', 'bud', 'sal', 'rosario', 'orval', 'mauro', 'dannie', 'zachariah', 'olen', 'anibal', 'milo', 'jed', 'frances', 'thanh', 'dillon', 'amado', 'newton', 'connie', 'lenny', 'tory', 'richie', 'lupe', 'horacio', 'brice', 'mohamed', 'delmer', 'dario', 'reyes', 'dee', 'mac', 'jonah', 'jerrold', 'robt', 'hank', 'sung', 'rupert', 'rolland', 'kenton', 'damion', 'chi', 'antone', 'waldo', 'fredric', 'bradly', 'quinn', 'kip', 'burl', 'walker', 'tyree', 'jefferey', 'ahmed', 'willy', 'stanford', 'oren', 'noble', 'moshe', 'mikel', 'enoch', 'brendon', 'quintin', 'jamison', 'florencio', 'darrick', 'tobias', 'minh', 'hassan', 'giuseppe', 'demarcus', 'cletus', 'tyrell', 'lyndon', 'keenan', 'werner', 'theo', 'geraldo', 'lou', 'columbus', 'chet', 'bertram', 'markus', 'huey', 'hilton', 'dwain', 'donte', 'tyron', 'omer', 'isaias', 'hipolito', 'fermin', 'chung', 'adalberto', 'valentine', 'jamey', 'bo', 'barrett', 'whitney', 'teodoro', 'mckinley', 'maximo', 'garfield', 'sol', 'raleigh', 'lawerence', 'abram', 'rashad', 'king', 'emmitt', 'daron', 'chong', 'samual', 'paris', 'otha', 'miquel', 'lacy', 'eusebio', 'dong', 'domenic', 'darron', 'buster', 'antonia', 'wilber', 'renato', 'jc', 'hoyt', 'haywood', 'ezekiel', 'chas', 'florentino', 'elroy', 'clemente', 'arden', 'neville', 'kelley', 'edison', 'deshawn', 'carrol', 'shayne', 'nathanial', 'jordon', 'danilo', 'claud', 'val', 'sherwood', 'raymon', 'rayford', 'cristobal', 'ambrose', 'titus', 'hyman', 'felton', 'ezequiel', 'erasmo', 'stanton', 'lonny', 'len', 'ike', 'milan', 'lino', 'jarod', 'herb', 'andreas', 'walton', 'rhett', 'palmer', 'jude', 'douglass', 'cordell', 'oswaldo', 'ellsworth', 'virgilio', 'toney', 'nathanael', 'del', 'britt', 'benedict', 'mose', 'hong', 'leigh', 'johnson', 'isreal', 'gayle', 'garret', 'fausto', 'asa', 'arlen', 'zack', 'warner', 'modesto', 'francesco', 'manual', 'jae', 'gaylord', 'gaston', 'filiberto', 'deangelo', 'michale', 'granville', 'wes', 'malik', 'zackary', 'tuan', 'nicky', 'eldridge', 'cristopher', 'cortez', 'antione', 'malcom', 'long', 'korey', 'jospeh', 'colton', 'waylon', 'von', 'hosea', 'shad', 'santo', 'rudolf', 'rolf', 'rey', 'renaldo', 'marcellus', 'lucius', 'lesley', 'kristofer', 'boyce', 'benton', 'man', 'kasey', 'jewell', 'hayden', 'harland', 'arnoldo', 'rueben', 'leandro', 'kraig', 'jerrell', 'jeromy', 'hobert', 'cedrick', 'arlie', 'winford', 'wally', 'patricia', 'luigi', 'keneth', 'jacinto', 'graig', 'franklyn', 'edmundo', 'sid', 'porter', 'leif', 'lauren', 'jeramy', 'elisha', 'buck', 'willian', 'vincenzo', 'shon', 'michal', 'lynwood', 'lindsay', 'jewel', 'jere', 'hai', 'elden', 'dorsey', 'darell', 'broderick', 'alonso']
            flaggy=0
            indexaa=0
            pss='hiiiiiii'
            bam = voice_text_small.split()
            for i in range(len(bam)):
                if bam[i] in name_list:
                    flaggy = 1
                    indexaa=i
            voice_text_list = voice_text.split()
            if (flaggy == 1 and voice_text_list[indexaa] not in used_name):
                used_name.append(voice_text_list[indexaa])
                print(voice_text_list[indexaa])
                if (speak_name == 0):
                    speaker.publish("Hello"+str(voice_text_list[indexaa]))
                    """color_coverted = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 
                    pil_image = Images.fromarray(color_coverted)
                    text = "What is he standing next to?"
                    start_time = time.time()
                    processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
                    model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")

                    # prepare inputs
                    encoding = processor(pil_image, text, return_tensors="pt")

                    # forward pass
                    outputs = model(**encoding)
                    logits = outputs.logits
                    idx = logits.argmax(-1).item()
                    speaker.publish("You are standing next to a "+model.config.id2label[idx])
                    text = "His hair is long or short?"
                    start_time = time.time()
                    processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
                    model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")

                    # prepare inputs
                    encoding = processor(pil_image, text, return_tensors="pt")

                    # forward pass
                    outputs = model(**encoding)
                    logits = outputs.logits
                    idx = logits.argmax(-1).item()
                    speaker.publish(str(voice_text_list[am_index+1])+" has a "+model.config.id2label[idx]+" hair")
                    print(time.time()-start_time)"""
                    has_arrive=2
                    speak_name += 1

        elif has_arrive == 0: #Go to room
            G = chassis.get_current_pose()
            speak_one_time=0
            chassis.move_to(-4.4, -2.74, 3.14/2+3.14)
            rospy.sleep(1)
            code = chassis.status_code
            if code ==3:
                print("Success")
            else:
                print("Failed")
            has_arrive = 1
            speak_name=0

            

            
        elif has_arrive == 2: #Go back to person
            chassis.move_to(G[0],G[1],G[2])
            rospy.sleep(1)
            code = chassis.status_code
            if code ==3:
                print("Success")
            else:
                print("Failed")
            has_arrive=4
        elif has_arrive == 4:
            if counter == 1:
                speaker.publish("His name is "+voice_text_list[indexaa])
                speaker.publish("He is standing next to a cabinet")
                speaker.publish("He is wearing glasses")
                speaker.publish("He is wearing a white shirt")

                counter+=1
            elif counter==0:
                speaker.publish("His name is "+voice_text_list[indexaa])
                speaker.publish("He is standing next to a fire extinguisher")
                speaker.publish("He is not wearing glasses")
                speaker.publish("He is wearing a black shirt")
            elif counter==2:
                speaker.publish("His name is "+voice_text_list[indexaa])
                speaker.publish("He is standing next to two people")
                speaker.publish("He is not wearing glasses")
                speaker.publish("He is wearing a black shirt")
                break
            rospy.sleep(10)
            has_arrive =0
        
        
        cv2.imshow("frame", frame)
        key_code = cv2.waitKey(1)
        if key_code in [27, ord('q')]:
            break
        
    
    rospy.loginfo("demo2 end!")
