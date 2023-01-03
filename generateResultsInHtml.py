import sys
import os

################################################################################
#Function: InitializeHTML
################################################################################
def InitializeHTML(tag):
  tag = "<!DOCTYPE html>\n"
  tag = tag + "<html>\n"
  tag = tag + "<head>\n"
  tag = tag + "<style>\n"
  tag = tag + "ul#menu {padding: 0;}\n"
  tag = tag + "ul#menu li a {background-color: white;color: black;padding: 10px 20px;text-decoration: none;}\n"
  tag = tag + "ul#menu li a:hover {background-color: orange;}\n"
  tag = tag + "</style>\n"
  tag = tag + "</head>\n"
  # tag = tag + "<body style=\"background:AntiqueWhite\"><font size=\"5\">CMYK Profiles:"
  tag = tag + "<body style=\"background:AntiqueWhite\"><font size=\"5\">Her2 Classification"
  return tag

################################################################################
#Function: FinalizeHTML
################################################################################
def FinalizeHTML(header, r_pass, r_fail, table, htmlFile):
  #Append the results to the the header

  header = header + "<ul id=\"menu\">\n"
  header = header + "<table style=\"width:100%\" border=\"1\">\n"
  #Append the table 
  header = header + table 
  #Close the html 
  header = header +"</table>\n"
  header = header +"</html>"
  
  #Save the html to a file 
  f = open(htmlFile, 'w')
  f.write(header)
  f.close()
  
################################################################################
#Function: PopulateResultTable
################################################################################
def PopulateResultTable(input_folder, dir,tableTag):

  input_folder_path = input_folder + dir + '/'
  input_list = os.listdir(input_folder_path)
  
  for i in range(len(input_list)):
    
    path1 = input_folder_path + input_list[i] 
    gt = input_list[i].split('_')[2]
    
    tableTag = tableTag + "<tr>\n"
    tableTag = tableTag + "<th>Image</th><th>Ground Truth</th><th>Prediction</th>"
    tableTag = tableTag + "<tr>\n"
                                                                                 
    if not os.path.exists(path1): print("Path not found : ",path1)

    print(path1)
      #Add the best focused image 
    tableTag = tableTag + "<td align=\"center\"><img src=\"file:///"
    tableTag = tableTag +path1+"\" alt=\"File Not Found\" style=\"width:55%;height:95%;border:solid\"></td>\n"
    
      # Add Ground Truth
    tableTag = tableTag + "<td align=\"center\">"
    tableTag = tableTag + gt
    tableTag = tableTag + "</td>\n"

    # Add prediction
    tableTag = tableTag + "<td align=\"center\">"
    tableTag = tableTag + dir
    tableTag = tableTag + "</td>\n"

  return tableTag
   
################################################################################
#Function: main()
#This is the main driver function for running the batch script on the data. It
#is used to branch off into multiple processes depending on what we seek to do.
#For example, if we want to rearrange the data, we branch off into the 
#Rerrange data process, else we use the regular autotest process.
################################################################################
def main():

  inputFolder = '/media/adminspin/558649a1-21fd-43b8-b53d-7334f802b47a/wsi_tb_data/Her2/code/output/jpeg/'
  summaryDir = '/media/adminspin/558649a1-21fd-43b8-b53d-7334f802b47a/wsi_tb_data/Her2/code/output/'
 
  # Check if the dummay directory exits
  if not os.path.exists(summaryDir):
    os.mkdir(summaryDir)
  
  r_pass = 0
  r_fail = 0
  htmlHeader = ""
  htmltable = ""
  
  # Initialize the header
  htmlHeader = InitializeHTML(htmlHeader)
  
  
  dir_list = os.listdir(inputFolder)
  
  htmltable = ""
  for dir in dir_list:
    htmltable = PopulateResultTable(inputFolder,dir,htmltable)
  filename  = 'IHC_Her2_Classification_predictions'
  summaryFileName = os.path.join(summaryDir, filename + ".html")
  FinalizeHTML(htmlHeader, r_pass, r_fail, htmltable, summaryFileName)
  
  exit()
  
if __name__ == "__main__":
    main()