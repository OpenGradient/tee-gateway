// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/// @dev Interface to the TEE verifier precompile at 0x0900 on OpenGradient.
interface ITEEVerifier {
    function verifyRSAPSS(
        bytes calldata publicKey,
        bytes32 messageHash,
        bytes calldata signature
    ) external view returns (bool);
}

/// @title Heartbeat — TEE-attested liveness pings verified via RSA-PSS precompile
///
/// Flow:
///   1. teeId = keccak256(publicKeyDER)  (derived from the TEE's RSA key)
///   2. TEE signs:  keccak256(abi.encodePacked(teeId, timestamp))  with RSA-PSS
///   3. Contract calls precompile to verify signature against the public key
///   4. Verifies keccak256(publicKey) == teeId  (binds key to identity)
///
/// First heartbeat for a teeId locks the public key; subsequent calls
/// check consistency automatically via the keccak256 derivation.
contract Heartbeat {
    ITEEVerifier constant TEE_VERIFIER =
        ITEEVerifier(0x0000000000000000000000000000000000000900);

    struct NodeInfo {
        bytes   publicKey;      // RSA public key DER (set on first heartbeat)
        uint256 lastHeartbeat;  // timestamp from the signed payload
        uint256 count;
    }

    mapping(bytes32 => NodeInfo) public nodes;

    event HeartbeatReceived(
        bytes32 indexed teeId,
        uint256 timestamp,
        uint256 count
    );

    /// @notice Submit a signed heartbeat from a TEE node.
    /// @param teeId      - keccak256(publicKey), uniquely identifies the TEE.
    /// @param timestamp   - Unix timestamp included in the signed message.
    /// @param publicKey   - RSA public key in DER format.
    /// @param signature   - RSA-PSS-SHA256 signature over keccak256(teeId ‖ timestamp).
    function heartbeat(
        bytes32 teeId,
        uint256 timestamp,
        bytes calldata publicKey,
        bytes calldata signature
    ) external {
        // Verify the public key matches the claimed teeId
        require(keccak256(publicKey) == teeId, "publicKey does not match teeId");

        // Build the message hash the TEE signed
        bytes32 messageHash = keccak256(abi.encodePacked(teeId, timestamp));

        // Verify RSA-PSS signature via the on-chain precompile
        bool valid = TEE_VERIFIER.verifyRSAPSS(publicKey, messageHash, signature);
        require(valid, "Invalid TEE signature");

        // Record heartbeat
        NodeInfo storage info = nodes[teeId];
        if (info.publicKey.length == 0) {
            info.publicKey = publicKey;
        }
        info.lastHeartbeat = timestamp;
        info.count += 1;

        emit HeartbeatReceived(teeId, timestamp, info.count);
    }

    /// @notice Look up a TEE node's liveness info.
    function getNodeInfo(bytes32 teeId)
        external
        view
        returns (bytes memory publicKey, uint256 lastHeartbeat, uint256 count)
    {
        NodeInfo storage info = nodes[teeId];
        return (info.publicKey, info.lastHeartbeat, info.count);
    }
}
