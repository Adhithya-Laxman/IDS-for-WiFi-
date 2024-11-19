package main
 
import (
	"fmt"
	"io"
	"net"
	"net/rpc"
	"os"
)
 
type Server struct{}
 
func (this *Server) SendFile(filename string, reply *[]byte) error {
	// Ensure the file exists in the current directory
	file, err := os.Open(filename)
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
 
func client() {
	c, err := rpc.Dial("tcp", "NotAvailable-33029.portmap.host:33029")
	if err != nil {
		fmt.Println("Connection error:", err)
		return
	}
 
	// Prepare a variable to hold the received file data
	var fileData []byte
 
	// Call the SendFile method on the server to receive "sample.txt"
	err = c.Call("Server.SendFile", "sample.txt", &fileData)
	if err != nil {
		fmt.Println("Error calling Server.SendFile:", err)
		return
	}
 
	// Write the received file data to a file in the current directory
	err = os.WriteFile("sample_received.txt", fileData, 0644)
	if err != nil {
		fmt.Println("Error writing file:", err)
		return
	}
 
	fmt.Println("File received and saved as sample_received.txt")
}
 
func main() {
	// go server()
	client()
}
