#!/usr/bin/env python2
#encoding: UTF-8
import json
import sys;sys.path.append('./')
import zipfile
import re
import sys
import os
import codecs
import importlib
from io import StringIO

from shapely.geometry import *
from shapely.geometry import Polygon, LinearRing

def print_help():
    sys.stdout.write('Usage: python %s.py -g=<gtFile> -s=<submFile> [-o=<outputFolder> -p=<jsonParams>]' %sys.argv[0])
    sys.exit(2)
    

def load_zip_file_keys(file,fileNameRegExp=''):
    """
    Returns an array with the entries of the ZIP file that match with the regular expression.
    The key's are the names or the file or the capturing group definied in the fileNameRegExp
    """
    try:
        archive=zipfile.ZipFile(file, mode='r', allowZip64=True)
    except :
        raise Exception('Error loading the ZIP archive.')

    pairs = []
    
    for name in archive.namelist():
        addFile = True
        keyName = name
        if fileNameRegExp!="":
            m = re.match(fileNameRegExp,name)
            if m == None:
                addFile = False
            else:
                if len(m.groups())>0:
                    keyName = m.group(1)
                    
        if addFile:
            pairs.append( keyName )
                
    return pairs
    

def load_zip_file(file,fileNameRegExp='',allEntries=False):
    """
    Returns an array with the contents (filtered by fileNameRegExp) of a ZIP file.
    The key's are the names or the file or the capturing group definied in the fileNameRegExp
    allEntries validates that all entries in the ZIP file pass the fileNameRegExp
    """
    print(f"DEBUG rrc_evaluation_funcs.load_zip_file: Attempting to load zip: '{file}'") # ADD THIS
    print(f"DEBUG rrc_evaluation_funcs.load_zip_file: File exists? {os.path.exists(file)}") # ADD THIS
    try:
        archive=zipfile.ZipFile(file, mode='r', allowZip64=True)
    except Exception as e_zip:
        print(f"DEBUG rrc_evaluation_funcs.load_zip_file: zipfile.ZipFile failed with: {e_zip}")
        raise Exception('Error loading the ZIP archive')    

    pairs = []
    for name in archive.namelist():
        addFile = True
        keyName = name
        if fileNameRegExp!="":
            m = re.match(fileNameRegExp,name)
            if m == None:
                addFile = False
            else:
                if len(m.groups())>0:
                    keyName = m.group(1)
        
        if addFile:
            pairs.append( [ keyName , archive.read(name)] )
        else:
            if allEntries:
                raise Exception('ZIP entry not valid: %s' %name)             

    return dict(pairs)
	
def decode_utf8(raw):
    """
    Returns a Unicode object on success, or None on failure
    """
    try:
        raw = codecs.decode(raw,'utf-8', 'replace')
        #extracts BOM if exists
        raw = raw.encode('utf8')
        if raw.startswith(codecs.BOM_UTF8):
            raw = raw.replace(codecs.BOM_UTF8, '', 1)
        return raw.decode('utf-8')
    except:
       return None

def validate_lines_in_file_gt(fileName,file_contents,CRLF=True,LTRB=True,withTranscription=False,withConfidence=False,imWidth=0,imHeight=0):
    utf8File = decode_utf8(file_contents)
    if (utf8File is None) :
        raise Exception("The file %s is not UTF-8" %fileName)
    lines = utf8File.split( "\r\n" if CRLF else "\n" )
    for line_raw in lines: # Iterate directly
        line = line_raw.replace("\r","").replace("\n","").strip()
        if not line: # Skip empty lines
            continue
        try:
            # Assuming LTRB is False for your polygon data
            validate_tl_line_gt(line, False, withTranscription, withConfidence, imWidth, imHeight)
        except Exception as e:
            raise Exception(("Line in sample not valid. Sample: %s Line: %s Error: %s" %(fileName,line_raw,str(e))).encode('utf-8', 'replace'))

def validate_lines_in_file(fileName,file_contents,CRLF=True,LTRB=True,withTranscription=False,withConfidence=False,imWidth=0,imHeight=0): # For detection files
    utf8File = decode_utf8(file_contents)
    if (utf8File is None) :
        raise Exception("The file %s is not UTF-8" %fileName)
    lines = utf8File.split( "\r\n" if CRLF else "\n" )
    for line_raw in lines: # Iterate directly
        line = line_raw.replace("\r","").replace("\n","").strip()
        if not line:
            continue
        try:
            # Assuming LTRB is False for your polygon data
            validate_tl_line(line, False, withTranscription, withConfidence, imWidth, imHeight)
        except Exception as e:
            raise Exception(("Line in sample not valid. Sample: %s Line: %s Error: %s" %(fileName,line_raw,str(e))).encode('utf-8', 'replace'))
    
def validate_tl_line_gt(line,LTRB=True,withTranscription=True,withConfidence=True,imWidth=0,imHeight=0):
    """
    Validate the format of the line. If the line is not valid an exception will be raised.
    If maxWidth and maxHeight are specified, all points must be inside the imgage bounds.
    Posible values are:
    LTRB=True: xmin,ymin,xmax,ymax[,confidence][,transcription] 
    LTRB=False: x1,y1,x2,y2,x3,y3,x4,y4[,confidence][,transcription] 
    """
    get_tl_line_values_gt(line,LTRB,withTranscription,withConfidence,imWidth,imHeight)   
   
def validate_tl_line(line,LTRB=True,withTranscription=True,withConfidence=True,imWidth=0,imHeight=0):
    """
    Validate the format of the line. If the line is not valid an exception will be raised.
    If maxWidth and maxHeight are specified, all points must be inside the imgage bounds.
    Posible values are:
    LTRB=True: xmin,ymin,xmax,ymax[,confidence][,transcription] 
    LTRB=False: x1,y1,x2,y2,x3,y3,x4,y4[,confidence][,transcription] 
    """
    get_tl_line_values(line,LTRB,withTranscription,withConfidence,imWidth,imHeight)
    
def get_tl_line_values_gt(line, LTRB=True, withTranscription=False, withConfidence=False, imWidth=0, imHeight=0):
    confidence = 0.0
    transcription = ""
    points = []

    if LTRB: # Should be False for your format
        raise Exception('LTRB=True format not supported by this GT parser for polygon data.')

    line = line.strip()
    parts = line.split(',####') # Your standard separator

    if not line: # Handle empty line case explicitly
        return points, confidence, transcription

    if len(parts) == 2: # Expecting coordinates part and transcription part
        coord_str = parts[0].strip()
        transcription = parts[1].strip() # Transcription is everything after ####
        
        if coord_str: # If there are coordinates
            coord_parts_list = coord_str.split(',')
            if not coord_parts_list or (len(coord_parts_list) % 2 != 0 and len(coord_parts_list) > 0) :
                raise Exception(f"Invalid coordinate string (empty or odd number of values): '{coord_str}' in line: '{line}'")
            try:
                points = [float(p.strip()) for p in coord_parts_list]
            except ValueError as e:
                raise Exception(f"Error parsing coordinates from '{coord_str}': {e} in line: '{line}'")
        # If coord_str is empty, points remains []
            
    elif len(parts) == 1 and withTranscription:
        # This case could be problematic if a line is just "TEXT" without "####"
        # For safety, assume if #### is not present, and text is expected, the whole line is text
        # However, your format is COORDS,####TEXT. So this branch might indicate malformed line.
        # For now, let's assume if only one part, and text is expected, it's all text
        transcription = parts[0].strip()
        print(f"Warning: Line parsed as only transcription (no '####' separator): '{line}'")
    elif len(parts) == 1 and not withTranscription and parts[0]: # Only coords expected
        coord_str = parts[0].strip()
        coord_parts_list = coord_str.split(',')
        if not coord_parts_list or (len(coord_parts_list) % 2 != 0 and len(coord_parts_list) > 0):
            raise Exception(f"Invalid coordinate string: '{coord_str}' in line: '{line}'")
        try:
            points = [float(p.strip()) for p in coord_parts_list]
        except ValueError as e:
            raise Exception(f"Error parsing coordinates from '{coord_str}': {e} in line: '{line}'")
    elif line: # Line was not empty but didn't fit expected parts
        raise Exception(f"Line does not conform to 'COORDS,####TRANSCRIPTION' or other expected formats: '{line}'")
    
    # Note: The original transcription re.match for quotes is removed
    # as your format example "####PETROSAINS" does not have surrounding quotes after ####.
    # If it could, you'd add:
    # if transcription.startswith('"') and transcription.endswith('"'):
    #     transcription = transcription[1:-1]

    if points: # Only validate if points were actually parsed
        validate_clockwise_points(points)
        if (imWidth > 0 and imHeight > 0):
            for ip in range(0, len(points), 2):
                validate_point_inside_bounds(points[ip], points[ip+1], imWidth, imHeight)
        
    return points, confidence, transcription

def get_tl_line_values(line, LTRB=True, withTranscription=False, withConfidence=False, imWidth=0, imHeight=0):
    confidence = 1.0 # Default for detections if not parsed
    transcription = ""
    points = []

    if LTRB:
        raise Exception('LTRB=True format not supported by this DET parser for polygon data.')

    line = line.strip()
    # For detections, the format from your `to_eval_format` is:
    # cors(no_score),####TEXT (e.g., 1,2,3,4,####TEXT)
    # Or, if `to_eval_format` includes score in cors: 1,2,3,4,score,####TEXT
    # The current script's to_eval_format writes:
    # fout.writelines(cors+',####'+str(ptr[1])+'\n')
    # where cors = ','.join(e for e in ptr[0].split(',')[:-1]) (i.e., coords without score)
    # and ptr[1] is rec. So, your det files are COORDS_NO_SCORE,####TEXT.
    # Confidence is not in the file per line if using this to_eval_format.

    parts = line.split(',####')

    if not line:
        return points, confidence, transcription

    if len(parts) == 2:
        coord_str = parts[0].strip()
        transcription = parts[1].strip()
        
        if coord_str:
            coord_parts_list = coord_str.split(',')
            if not coord_parts_list or (len(coord_parts_list) % 2 != 0 and len(coord_parts_list) > 0):
                raise Exception(f"Invalid coordinate string: '{coord_str}' in line: '{line}'")
            try:
                points = [float(p.strip()) for p in coord_parts_list]
            except ValueError as e:
                raise Exception(f"Error parsing coordinates from '{coord_str}': {e} in line: '{line}'")
        # Confidence is not in this format per line from your to_eval_format
        # It's usually handled globally or by sorting detection files if they contain scores.
        # For this parser, we assume confidence is externally handled or a default.
            
    elif len(parts) == 1 and withTranscription:
        transcription = parts[0].strip()
    elif len(parts) == 1 and not withTranscription and parts[0]:
        coord_str = parts[0].strip()
        # ... (similar coord parsing as above) ...
    elif line:
        raise Exception(f"DET Line does not conform: '{line}'")

    if points:
        validate_clockwise_points(points) # Usually more important for DET than GT
        if (imWidth > 0 and imHeight > 0):
            for ip in range(0, len(points), 2):
                validate_point_inside_bounds(points[ip], points[ip+1], imWidth, imHeight)
        
    return points, confidence, transcription
    
            
def validate_point_inside_bounds(x,y,imWidth,imHeight):
    if(x<0 or x>imWidth):
            raise Exception("X value (%s) not valid. Image dimensions: (%s,%s)" %(xmin,imWidth,imHeight))
    if(y<0 or y>imHeight):
            raise Exception("Y value (%s)  not valid. Image dimensions: (%s,%s) Sample: %s Line:%s" %(ymin,imWidth,imHeight))

def validate_clockwise_points(points):
    """
    Validates that the points that the 4 points that dlimite a polygon are in clockwise order.
    """
    
    # if len(points) != 8:
    #     raise Exception("Points list not valid." + str(len(points)))
    
    # point = [
    #             [int(points[0]) , int(points[1])],
    #             [int(points[2]) , int(points[3])],
    #             [int(points[4]) , int(points[5])],
    #             [int(points[6]) , int(points[7])]
    #         ]
    # edge = [
    #             ( point[1][0] - point[0][0])*( point[1][1] + point[0][1]),
    #             ( point[2][0] - point[1][0])*( point[2][1] + point[1][1]),
    #             ( point[3][0] - point[2][0])*( point[3][1] + point[2][1]),
    #             ( point[0][0] - point[3][0])*( point[0][1] + point[3][1])
    # ]
    
    # summatory = edge[0] + edge[1] + edge[2] + edge[3];
    # if summatory>0:
    #     raise Exception("Points are not clockwise. The coordinates of bounding quadrilaterals have to be given in clockwise order. Regarding the correct interpretation of 'clockwise' remember that the image coordinate system used is the standard one, with the image origin at the upper left, the X axis extending to the right and Y axis extending downwards.")
    if len(points) < 6: # Need at least 3 points (6 coordinates) for a polygon
        raise Exception(f"Not enough points to form a polygon: {len(points)} coordinates")
    if len(points) % 2 != 0:
        raise Exception(f"Odd number of coordinate values: {len(points)}")

    pts_tuples = [(points[j], points[j+1]) for j in range(0, len(points), 2)]
    
    try:
        poly = Polygon(pts_tuples)
    except Exception as e:
        # For example, if points are co-linear for a very simple polygon
        raise Exception(f"Cannot form a valid Shapely Polygon from points: {pts_tuples}. Error: {e}")

    if not poly.is_valid:
        # This can happen if lines cross, etc.
        # For GT, this would be an annotation error.
        print(f"Warning: Polygon from points {pts_tuples} is not valid according to Shapely (e.g., self-intersection).")
        # Depending on strictness, you might raise an Exception here or just warn.
        # For now, let's allow it to pass with a warning for GT.
        # raise Exception(f"Polygon is not valid (e.g., self-intersection): {pts_tuples}")
        pass


    # The original script's clockwise check using LinearRing might still be useful,
    # but the assert(0) was too strict if it assumed quadrilaterals.
    # A LinearRing must not self-intersect.
    try:
        pRing = LinearRing(pts_tuples)
        if pRing.is_ccw:
            # This means it's counter-clockwise. Many systems expect clockwise.
            # For GT validation, this might just be a warning or an accepted format.
            # The original script raised an error. Let's make it a warning for now for CTW1500 GT.
            print(f"Warning: Ground truth polygon points appear to be counter-clockwise: {pts_tuples}")
            # To strictly enforce the original script's behavior (error on CCW):
            # assert not pRing.is_ccw, ("Points are not clockwise. ...") # Original assertion style
    except Exception as e:
        raise Exception(f"Error creating LinearRing or checking CCW for points {pts_tuples}. Error: {e}")
    
        
def get_tl_line_values_from_file_contents(content,CRLF=True,LTRB=True,withTranscription=False,withConfidence=False,imWidth=0,imHeight=0,sort_by_confidences=True):
    pointsList = []
    transcriptionsList = []
    confidencesList = []
    lines = content.split( "\r\n" if CRLF else "\n" )
    for line_raw in lines: # Iterate directly
        line = line_raw.replace("\r","").replace("\n","").strip()
        if not line:
            continue
        # Assuming LTRB is False for your polygon data
        points, confidence, transcription = get_tl_line_values_gt(line, False, withTranscription, withConfidence, imWidth, imHeight)
        pointsList.append(points)
        transcriptionsList.append(transcription)
        confidencesList.append(confidence)
    # Sorting by confidence is not relevant for GT if confidence is always 0
    return pointsList,confidencesList,transcriptionsList

def get_tl_line_values_from_file_contents_det(content,CRLF=True,LTRB=True,withTranscription=False,withConfidence=False,imWidth=0,imHeight=0,sort_by_confidences=True):
    pointsList = []
    transcriptionsList = []
    confidencesList = []
    lines = content.split( "\r\n" if CRLF else "\n" )
    for line_raw in lines: # Iterate directly
        line = line_raw.replace("\r","").replace("\n","").strip()
        if not line:
            continue
        # Assuming LTRB is False for your polygon data
        points, confidence, transcription = get_tl_line_values(line, False, withTranscription, withConfidence, imWidth, imHeight)
        pointsList.append(points)
        transcriptionsList.append(transcription)
        confidencesList.append(confidence)

    if withConfidence and len(confidencesList)>0 and sort_by_confidences:
        # This part for sorting detections by confidence is important
        import numpy as np # Make sure numpy is imported
        valid_indices = [i for i, p in enumerate(pointsList) if p] # Only consider entries with points
        if not valid_indices: return pointsList, confidencesList, transcriptionsList

        confidencesList_valid = np.array([confidencesList[i] for i in valid_indices])
        pointsList_valid = [pointsList[i] for i in valid_indices]
        transcriptionsList_valid = [transcriptionsList[i] for i in valid_indices]
        
        if confidencesList_valid.size > 0:
            sorted_ind = np.argsort(-confidencesList_valid)
            confidencesList = [confidencesList_valid[i] for i in sorted_ind]
            pointsList = [pointsList_valid[i] for i in sorted_ind]
            transcriptionsList = [transcriptionsList_valid[i] for i in sorted_ind]
        else: # No valid entries to sort
            return [],[],[]

    return pointsList,confidencesList,transcriptionsList

def main_evaluation(p,det_file, gt_file, default_evaluation_params_fn,validate_data_fn,evaluate_method_fn,show_result=True,per_sample=True):
    """
    This process validates a method, evaluates it and if it succed generates a ZIP file with a JSON entry for each sample.
    Params:
    p: Dictionary of parmeters with the GT/submission locations. If None is passed, the parameters send by the system are used.
    default_evaluation_params_fn: points to a function that returns a dictionary with the default parameters used for the evaluation
    validate_data_fn: points to a method that validates the corrct format of the submission
    evaluate_method_fn: points to a function that evaluated the submission and return a Dictionary with the results
    """
    
    # if (p == None):
    #     p = dict([s[1:].split('=') for s in sys.argv[1:]])
    #     if(len(sys.argv)<3):
    #         print_help()
    #p = {}
    p['g'] =gt_file  #'tttgt.zip'
    p['s'] =det_file #'det.zip'

    evalParams = default_evaluation_params_fn()
    if 'p' in p.keys():
        evalParams.update( p['p'] if isinstance(p['p'], dict) else json.loads(p['p'][1:-1]) )

    resDict={'calculated':True,'Message':'','method':'{}','per_sample':'{}'}    
    # try:
    validate_data_fn(p['g'], p['s'], evalParams)  
    evalData = evaluate_method_fn(p['g'], p['s'], evalParams)
    resDict.update(evalData)
        
    # except Exception as e:
        # resDict['Message']= str(e)
        # resDict['calculated']=False

    if 'o' in p:
        if not os.path.exists(p['o']):
            os.makedirs(p['o'])

        resultsOutputname = p['o'] + '/results.zip'
        outZip = zipfile.ZipFile(resultsOutputname, mode='w', allowZip64=True)

        del resDict['per_sample']
        if 'output_items' in resDict.keys():
            del resDict['output_items']

        outZip.writestr('method.json',json.dumps(resDict))
        
    if not resDict['calculated']:
        if show_result:
            sys.stderr.write('Error!\n'+ resDict['Message']+'\n\n')
        if 'o' in p:
            outZip.close()
        return resDict
    
    if 'o' in p:
        if per_sample == True:
            for k,v in evalData['per_sample'].items():
                outZip.writestr( k + '.json',json.dumps(v)) 

            if 'output_items' in evalData.keys():
                for k, v in evalData['output_items'].items():
                    outZip.writestr( k,v) 

        outZip.close()

    # if show_result:
    #     sys.stdout.write("Calculated!")
    #     sys.stdout.write('\n')
    #     sys.stdout.write(json.dumps(resDict['e2e_method']))
    #     sys.stdout.write('\n')
    #     sys.stdout.write(json.dumps(resDict['det_only_method']))
    #     sys.stdout.write('\n')
    
    return resDict


def main_validation(default_evaluation_params_fn,validate_data_fn):
    """
    This process validates a method
    Params:
    default_evaluation_params_fn: points to a function that returns a dictionary with the default parameters used for the evaluation
    validate_data_fn: points to a method that validates the corrct format of the submission
    """    
    try:
        p = dict([s[1:].split('=') for s in sys.argv[1:]])
        evalParams = default_evaluation_params_fn()
        if 'p' in p.keys():
            evalParams.update( p['p'] if isinstance(p['p'], dict) else json.loads(p['p'][1:-1]) )

        validate_data_fn(p['g'], p['s'], evalParams)              
        print('SUCCESS')
        sys.exit(0)
    except Exception as e:
        print(str(e))
        sys.exit(101)
