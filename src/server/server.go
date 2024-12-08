package main

import (
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
	id                string
	name              string
	instances_updated int
	content           []byte
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
	currentTime := time.Now().Format(time.RFC3339)
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
		fmt.Print(id, list.Valid[i])
		if id == list.Valid[i] {
			return true
		}
	}

	return false
}

func (this *Server) SendFile(id string, reply *[]byte) error {
	// Ensure the file exists in the current directory

	fmt.Println(id, len(id))

	if !checkAuth(id) {
		return fmt.Errorf("Invalid member!")
	}

	path := os.Getenv("PARAMS_PATH")

	file, err := os.Open(path)

	if err != nil {
		return err
	}
	defer file.Close()

	// Read the file content
	content, err := io.ReadAll(file)
	if err != nil {
		return err
	}

	// Set the reply to the file content
	*reply = content
	return nil
}

func (this *Server) SaveFile(args Args, reply string) error {
	fmt.Print(args.id)

	if !checkAuth(args.id) {

		return fmt.Errorf("Invalid member!")
	}

	path := "./trained models/" + args.name + ".pth"
	err := os.WriteFile(path, args.content, 0644)

	if err != nil {
		return err
	}

	var response Response
	jsonData, _ := os.ReadFile("../FEDavg/client_configuration.json")
	NewClient := []Client{{args.name, args.name, args.instances_updated, path, ""}}
	json.Unmarshal(jsonData, &response)
	updateData(&response, NewClient)
	writeDataToFile(&response, "../FEDavg/client_configuration.json")

	exec.Command("source /home/karthikeyan/.local/share/pipx/venvs/torch/bin/activate")
	exec.Command("python3 ../FEDavg/fedavg.py")
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
		fmt.Print("Accepted!")
		go rpc.ServeConn(c)
	}
}

func main() {
	server()
}
