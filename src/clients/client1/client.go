package main

import (
	"flag"
	"fmt"
	"io"
	"net/rpc"
	"os"
	"os/exec"
	"strconv"
)

type Args struct {
	id                string
	name              string
	instances_updated int
	content           []byte
}

func getParams() {
	c, err := rpc.Dial("tcp", "NotAvailable-37552.portmap.host:37552")
	path := os.Getenv("PARAMS_PATH")
	id := os.Getenv("UNIQUE_ID")
	fmt.Println(id)

	if err != nil {
		fmt.Println("Connection error:", err)
		return
	}

	// Prepare a variable to hold the received file data
	var fileData []byte

	err = c.Call("Server.SendFile", id, &fileData)
	if err != nil {
		fmt.Println("Error calling Server.SendFile:", err)
		return
	}

	// Write the received file data to a file in the current directory
	err = os.WriteFile(path, fileData, 0644)

	if err != nil {
		fmt.Println("Error writing file:", err)
		return
	}

	fmt.Println("Params recieved")
	cmd1 := exec.Command("ls", "-l")
	cmd1.Stdout = os.Stdout // Ensure output is printed to the terminal
	cmd1.Stderr = os.Stderr
	if err := cmd1.Run(); err != nil {
		fmt.Println("Error executing 'ls -l':", err)
	}

	// Execute the Python script
	cmd2 := exec.Command("python3", "../../models/DBN_client.py")
	cmd2.Stdout = os.Stdout // Ensure output is printed to the terminal
	cmd2.Stderr = os.Stderr
	if err := cmd2.Run(); err != nil {
		fmt.Println("Error executing Python script:", err)
	}
}

func updateParams() {
	path := os.Getenv("PARAMS_PATH")
	id := os.Getenv("UNIQUE_ID")
	name := os.Getenv("NAME")
	no := os.Getenv("NUMBER_ITR")
	c, err := rpc.Dial("tcp", "NotAvailable-37552.portmap.host:37552")

	if err != nil {
		return
	}

	file, err := os.Open(path)

	if err != nil {
		return
	}
	defer file.Close()

	// Read the file content
	content, err := io.ReadAll(file)

	val, _ := strconv.Atoi(no)

	args := Args{id, name, val, content}

	if err != nil {
		return
	}

	err = c.Call("Server.SaveFile", args, "")
	if err != nil {
		fmt.Println("Error calling Server.SendFile:", err)
		return
	}
}

func client() {
	call := flag.String("opn", "", "Specify the mode \nupdt - sends the params to server\ndwn - downloads the params from the server\n")
	flag.Parse()

	switch *call {
	case "updt":
		updateParams()
	case "dwn":
		getParams()
	default:
		return
	}
}

func main() {
	client()
}
