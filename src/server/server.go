package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net"
	"net/rpc"
	"os"
	"os/exec"
	"time"
)

type Server struct{} 

type UserList struct {
	Valid []string `json:"valid"`
}

type Args struct {
	Id                string
	Name              string
	Instances_updated int
	Content           []byte
}

type Client struct {
	ClientID                      string `json:"client_id"`
	Name                          string `json:"name"`
	InstancesTrainedLastIteration int    `json:"instances_trained_last_iteration"`
	ModelPath                     string `json:"model_path"`
	LastUpdated                   string `json:"last_updated"`
}

type LastIterationSummary struct {
	TotalInstancesTrained int    `json:"total_instances_trained"`
	GlobalModelPath       string `json:"global_model_path"`
	LastUpdated           string `json:"last_updated"`
}

type NetworkSummary struct {
	TotalClients         int                  `json:"total_clients"`
	Timestamp            string               `json:"timestamp"`
	Clients              []Client             `json:"clients"`
	LastIterationSummary LastIterationSummary `json:"last_iteration_summary"`
}

type Response struct {
	NetworkSummary NetworkSummary `json:"network_summary"`
}

// Update JSON data with new clients and update instances
func updateData(existingData *Response, newClients []Client) error {
	// Append the new clients to the existing client list
	existingData.NetworkSummary.Clients = append(existingData.NetworkSummary.Clients, newClients...)

	// Update total instances trained
	totalInstances := 0
	for _, client := range existingData.NetworkSummary.Clients {
		totalInstances += client.InstancesTrainedLastIteration
	}
	existingData.NetworkSummary.LastIterationSummary.TotalInstancesTrained = totalInstances

	// Update the timestamp and last_updated
	currentTime := time.Now().UTC().Format("2006-01-02T15:04:05Z")
	existingData.NetworkSummary.Timestamp = currentTime
	existingData.NetworkSummary.LastIterationSummary.LastUpdated = currentTime

	return nil
}

// Write updated data to a file
func writeDataToFile(data *Response, fileName string) error {
	// Open the file for writing (create if doesn't exist)
	file, err := os.Create(fileName)
	if err != nil {
		return fmt.Errorf("error creating file: %v", err)
	}
	defer file.Close()

	// Marshal the data into JSON format
	updatedData, err := json.MarshalIndent(data, "", "  ")
	if err != nil {
		return fmt.Errorf("error marshalling data to JSON: %v", err)
	}

	// Write the updated JSON data to the file
	_, err = file.Write(updatedData)
	if err != nil {
		return fmt.Errorf("error writing data to file: %v", err)
	}

	return nil
}

func checkAuth(id string) bool {
	var list UserList

	path := os.Getenv("AUTH_JSON")
	jsonData, err := os.ReadFile(path)

	fmt.Print(path)

	if err != nil {
		fmt.Print(err)
	}

	err = json.Unmarshal([]byte(jsonData), &list)

	if err != nil {
		fmt.Println(err)
	}

	for i := 0; i < len(list.Valid); i++ {
		if id == list.Valid[i] {
			return true
		}
	}

	return false
}

func (this *Server) SendFile(id string, reply *[]byte) error {
	// Ensure the file exists in the current directory
	fmt.Print(id, "is downloading the params.")

	if !checkAuth(id) {
		return fmt.Errorf("Invalid member!")
	}

	path := os.Getenv("PARAMS_PATH")

	file, err := os.Open(path)

	if err != nil {
		return err
	}
	defer file.Close() //close whenever an error/exception occurs

	// Read the file content
	content, err := io.ReadAll(file)
	if err != nil {
		return err
	}

	// Set the reply to the file content
	*reply = content
	return nil
}

func (this *Server) SaveFile(args Args, reply *string) error {
	fmt.Print(args.Id, "is Updating the params.")

	if !checkAuth(args.Id) {

		return fmt.Errorf("Invalid member!")
	}

	path := os.Getenv("HOME") + "/IDS-for-WiFi-/src/trained models/" + args.Name + ".pth"
	err := os.WriteFile(path, args.Content, 0644)

	if err != nil {
		return err
	}

	var response Response
	jsonData, err := os.ReadFile(os.Getenv("HOME") + "/IDS-for-WiFi-/src/FEDavg/client_configuration.json")
	fmt.Print(err)
	NewClient := []Client{{args.Id, args.Name, args.Instances_updated, path, time.Now().UTC().Format("2006-01-02T15:04:05Z")}}
	json.Unmarshal(jsonData, &response)
	updateData(&response, NewClient)
	writeDataToFile(&response, os.Getenv("HOME")+"/IDS-for-WiFi-/src/FEDavg/client_configuration.json")

	fmt.Println("Params recieved")
	var out bytes.Buffer
	cmd := exec.Command(os.Getenv("HOME") + "/IDS-for-WiFi-/src/FEDavg/fedavg.py")
	cmd.Stdout = &out
	fmt.Print(cmd.Run())
	fmt.Print(out.String())
	return nil
}

func server() {
	rpc.Register(new(Server))
	ln, err := net.Listen("tcp", "0.0.0.0:8080")

	if err != nil {
		fmt.Println(err)
		return
	}
	for {
		c, err := ln.Accept()
		if err != nil {
			continue
		}
		go rpc.ServeConn(c)
	}
}

func main() {
	server()
}
